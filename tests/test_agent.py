from __future__ import annotations

import copy
import unittest

import numpy as np

from agent import (
    COMPLETION_ITEMS,
    MISTAKE_ITEMS,
    ChunkAssembler,
    LanguageTutor,
    contains_expected,
    parse_llm_json,
    phoneme_score,
    pseudo_phonemes,
    similarity,
)


class ScriptedLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = []

    def complete(self, system: str, user: str, max_tokens: int = 220) -> str:
        self.calls.append((system, user, max_tokens))
        return self.payload


class BoomLLM:
    def complete(self, *args, **kwargs):
        raise RuntimeError("vLLM down")


class TutorTests(unittest.TestCase):
    def setUp(self):
        self.tutor = LanguageTutor(llm=None, stt=None, tts=None)
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])

    def test_curriculum_is_kid_safe_and_complete(self):
        for item in COMPLETION_ITEMS:
            self.assertTrue(item["stem"])
            self.assertTrue(item["expected"])
            self.assertIn(item["expected"].lower(), item["full"].lower())
            self.assertLessEqual(len(item["full"].split()), 12)
        for item in MISTAKE_ITEMS:
            self.assertNotEqual(item["spoken"], item["correct"])
            self.assertIn(item["kind"], {"grammar", "pronunciation"})
            self.assertTrue(item["hint"])

    def test_contains_expected_allows_little_words(self):
        self.assertTrue(contains_expected("the food bowl!", "the food bowl"))
        self.assertTrue(contains_expected("I think it is raining", "raining"))
        self.assertFalse(contains_expected("banana", "the food bowl"))

    def test_phonemes_treat_see_and_sea_as_close(self):
        self.assertGreaterEqual(phoneme_score("see", "sea"), 0.99)
        self.assertLess(phoneme_score("free cookies", "three cookies"), 1.0)
        self.assertIn("T", pseudo_phonemes("three"))

    def test_parse_llm_json_from_fences_and_prose(self):
        self.assertTrue(parse_llm_json('```json\n{"correct": true, "speak": "Nice!"}\n```')["correct"])
        self.assertIn("speak", parse_llm_json("Teddy says hello"))

    def test_completion_accepts_ending_and_advances(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        first = self.tutor.current_item()["id"]
        self.tutor.handle_transcript("the food bowl")
        self.assertNotEqual(self.tutor.current_item()["id"], first)
        self.assertEqual(self.tutor.state.streak, 1)
        self.assertEqual(self.tutor.state.turns, 1)

    def test_completion_rejects_unrelated_words_and_keeps_item(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        item_id = self.tutor.current_item()["id"]
        self.tutor.handle_transcript("spaceship lasagna")
        self.assertEqual(self.tutor.current_item()["id"], item_id)
        self.assertEqual(self.tutor.state.streak, 0)

    def test_mistake_accepts_correction(self):
        self.tutor.set_mode("mistake")
        self.tutor.item = copy.deepcopy(MISTAKE_ITEMS[0])
        events = self.tutor.handle_transcript("She doesn't like apples.")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertTrue(ui.payload["correct"])
        self.assertEqual(self.tutor.state.streak, 1)

    def test_mistake_flags_repeated_error(self):
        self.tutor.set_mode("mistake")
        self.tutor.item = copy.deepcopy(MISTAKE_ITEMS[0])
        events = self.tutor.handle_transcript("She don't like apples.")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertFalse(ui.payload["correct"])
        self.assertIn(ui.payload["issue"], {"grammar", "pronunciation"})

    def test_pronunciation_item_rewards_th_sound(self):
        self.tutor.set_mode("mistake")
        self.tutor.item = copy.deepcopy(next(x for x in MISTAKE_ITEMS if x["id"] == "three"))
        events = self.tutor.handle_transcript("I have three cookies")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertTrue(ui.payload["correct"])

    def test_llm_overlay_keeps_cheer_text(self):
        llm = ScriptedLLM('{"speak": "What a star!", "feedback": "Custom cheer"}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = tutor.handle_transcript("the food bowl")
        texts = " ".join(str(e.payload) for e in events)
        self.assertIn("What a star", texts)
        self.assertTrue(llm.calls)

    def test_llm_can_invent_next_item(self):
        llm = ScriptedLLM(
            '{"correct": true, "speak": "Yes! The red kite flew over", "feedback": "Nice", '
            '"next": {"id": "kite", "stem": "The red kite flew over", "expected": "the hill", '
            '"full": "The red kite flew over the hill."}}'
        )
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        tutor.handle_transcript("the food bowl")
        self.assertEqual(tutor.current_item()["id"], "kite")

    def test_llm_failure_falls_back_to_rules(self):
        tutor = LanguageTutor(llm=BoomLLM(), stt=None, tts=None)
        tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = tutor.handle_transcript("the food bowl")
        self.assertTrue(any(e.type == "ui" and e.payload.get("correct") is True for e in events))

    def test_random_opener_is_from_bank(self):
        self.tutor.set_mode("complete")
        ids = {self.tutor.pick_random_item()["id"] for _ in range(12)}
        self.assertTrue(ids.issubset({item["id"] for item in COMPLETION_ITEMS}))
        self.assertGreaterEqual(len(ids), 2)

    def test_chunk_assembler_emits_after_silence(self):
        asm = ChunkAssembler(sample_rate=16000, silence_rms=0.05, silence_ms=250, min_speech_ms=100)
        speech = np.ones(1600, dtype=np.float32) * 0.2
        quiet = np.zeros(8000, dtype=np.float32)
        self.assertIsNone(asm.push(speech))
        out = asm.push(quiet)
        self.assertIsNotNone(out)
        self.assertEqual(len(out), len(speech) + len(quiet))

    def test_ingest_ignored_until_ready(self):
        speech = np.ones(4000, dtype=np.float32) * 0.2
        self.tutor.accepting = False
        self.assertIsNone(self.tutor.take_utterance(speech, force=True))
        self.tutor.mark_ready()
        self.assertIsNotNone(self.tutor.take_utterance(speech, force=True))

    def test_empty_transcript_does_not_advance(self):
        item_id = self.tutor.current_item()["id"]
        self.tutor.handle_transcript("   ")
        self.assertEqual(self.tutor.current_item()["id"], item_id)

    def test_mode_switch_resets_streak(self):
        self.tutor.handle_transcript("the food bowl")
        self.assertEqual(self.tutor.state.streak, 1)
        self.tutor.set_mode("mistake")
        self.assertEqual(self.tutor.state.streak, 0)
        self.assertEqual(self.tutor.state.mode, "mistake")

    def test_similarity_bounds(self):
        self.assertEqual(similarity("a", "a"), 1.0)
        self.assertEqual(similarity("", "x"), 0.0)


if __name__ == "__main__":
    unittest.main()
