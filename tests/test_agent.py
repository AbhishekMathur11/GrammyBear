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
    sanitize_name,
    similarity,
    split_speech_chunks,
)


class ScriptedLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = []

    def complete(self, system: str, user: str, max_tokens: int = 220, temperature: float = 0.85, **kwargs) -> str:
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
        self.assertEqual(sanitize_name(""), "friend")
        self.assertEqual(sanitize_name("  sam!!  "), "Sam")

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
        self.assertEqual(self.tutor.state.turns, 1)
        self.assertNotIn("food bowl", self.tutor._pending_line.lower())
        self.assertNotIn("hoping", self.tutor._pending_line.lower())

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
        self.assertNotIn("she doesn't like apples", self.tutor._pending_line.lower())

    def test_pronunciation_item_rewards_th_sound(self):
        self.tutor.set_mode("mistake")
        self.tutor.item = copy.deepcopy(next(x for x in MISTAKE_ITEMS if x["id"] == "three"))
        events = self.tutor.handle_transcript("I have three cookies")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertTrue(ui.payload["correct"])

    def test_llm_invents_next_prompt(self):
        llm = ScriptedLLM('{"stem": "The tiny ant crawled under", "expected": "the leaf"}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = tutor.handle_transcript("the food bowl")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertTrue(ui.payload["correct"])
        self.assertIn("Great job, friend", tutor._pending_line)
        self.assertEqual(tutor.current_item()["expected"], "the leaf")
        self.assertTrue(llm.calls)
        texts = [event.payload["text"] for event in events if event.type == "speak_text"]
        self.assertGreaterEqual(len(texts), 1)
        combined = " ".join(texts).lower()
        self.assertIn("next one", combined)
        self.assertIn("tiny ant", combined)

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

    def test_generated_items_stay_unique(self):
        self.tutor.set_mode("complete")
        stems = []
        for _ in range(8):
            stems.append(self.tutor.invent_item()["stem"])
        self.assertGreaterEqual(len(set(stems)), 6)

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

    def test_garbled_audio_asks_to_repeat(self):
        item_id = self.tutor.current_item()["id"]
        events = self.tutor.handle_transcript("uh")
        line = self.tutor._pending_line.lower()
        self.assertTrue(any(word in line for word in ("catch", "fuzzy", "missed", "repeat", "say")))
        self.assertEqual(self.tutor.current_item()["id"], item_id)
        self.assertEqual(self.tutor.state.streak, 0)
        self.assertEqual(self.tutor.state.turns, 0)

    def test_llm_is_the_only_judge_for_completion(self):
        llm = ScriptedLLM('{"correct": false}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = tutor.handle_transcript("the food bowl")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertFalse(ui.payload["correct"])
        self.assertEqual(tutor.state.streak, 0)

    def test_llm_is_the_only_judge_for_story_synonyms(self):
        llm = ScriptedLLM('{"correct": false}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.set_mode("story")
        tutor.item = {
            "id": "happy",
            "sentence": "The happy puppy ran to the park.",
            "target": "happy",
            "speak": "The happy puppy ran to the park. What's another word for happy?",
            "stem": "The happy puppy ran to the park.",
            "expected": "happy",
        }
        events = tutor.handle_transcript("spaceship")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertFalse(ui.payload["correct"])

    def test_completion_cheer_is_short_celebration(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = self.tutor.handle_transcript("the food bowl")
        ui = next(e for e in events if e.type == "ui" and e.payload.get("correct") is True)
        self.assertIn(ui.payload["feedback"], {"You are awesome!", "Way to go!", "Amazing!"})

    def test_llm_cannot_praise_a_wrong_answer(self):
        llm = ScriptedLLM('{"correct": false}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = tutor.handle_transcript("spaceship lasagna")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertFalse(ui.payload["correct"])
        self.assertNotIn("great job", tutor._pending_line.lower())
        self.assertNotIn("food bowl", tutor._pending_line.lower())

    def test_empty_transcript_does_not_advance(self):
        item_id = self.tutor.current_item()["id"]
        self.tutor.handle_transcript("   ")
        self.assertEqual(self.tutor.current_item()["id"], item_id)

    def test_mode_switch_resets_streak(self):
        self.tutor.handle_transcript("the food bowl")
        self.assertEqual(self.tutor.state.streak, 1)
        self.tutor.set_mode("mistake")
        self.assertEqual(self.tutor.state.streak, 0)
        self.assertEqual(self.tutor.state.best, 1)
        self.assertEqual(self.tutor.state.mode, "mistake")

    def test_spoken_reply_stays_short(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        self.tutor.handle_transcript("the food bowl")
        self.assertIn("Great job", self.tutor._pending_line)
        self.assertNotIn("can you finish that sentence", self.tutor._pending_line.lower())

    def test_intro_happens_once_and_explains_each_game(self):
        self.tutor.configure(name="Sam", voice="af_bella")
        self.tutor.start_turn()
        first = self.tutor._pending_line
        self.assertIn("I'm Bella", first)
        self.assertIn("Finish the Sentence", first)
        self.assertIn("Sam", first)
        self.tutor.set_mode("mistake")
        self.tutor.start_turn()
        second = self.tutor._pending_line
        self.assertNotIn("I'm Bella", second)
        self.assertIn("Catch the Mistake", second)

    def test_switching_modes_explains_again(self):
        self.tutor.configure(name="Sam", voice="af_bella")
        self.tutor.set_mode("complete")
        self.tutor.start_turn()
        self.tutor.set_mode("mistake")
        self.tutor.start_turn()
        self.assertIn("Catch the Mistake", self.tutor._pending_line)
        self.tutor.set_mode("complete")
        self.tutor.start_turn()
        self.assertIn("Finish the Sentence", self.tutor._pending_line)
        self.assertNotIn("I'm Bella", self.tutor._pending_line)

    def test_story_opening_sounds_like_story_time(self):
        self.tutor.configure(name="Sam", voice="af_bella")
        self.tutor.set_mode("story")
        self.tutor.start_turn()
        line = self.tutor._pending_line.lower()
        self.assertIn("another word", line)
        self.assertIn("guess the synonym", line)
        self.assertNotIn("complete sentences", line)
        self.assertNotIn("write a story", line)

    def test_llm_accepts_another_valid_ending(self):
        llm = ScriptedLLM('{"correct": true}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.item = {
            "id": "treat",
            "stem": "The cat saw the yummy treat on the",
            "expected": "kitchen counter",
            "full": "The cat saw the yummy treat on the kitchen counter.",
        }
        events = tutor.handle_transcript("table")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertTrue(ui.payload["correct"])
        self.assertIn("indeed table", tutor._pending_line.lower())

    def test_speech_chunks_pause_between_sentences(self):
        chunks = split_speech_chunks("Great job, Sam! It is indeed moon. The sleepy kitten hid behind the")
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].endswith("!"))

    def test_praise_uses_child_name_not_the_answer(self):
        self.tutor.configure(name="Sam", voice="af_sky")
        self.tutor.set_mode("complete")
        self.tutor.configure(name="Sam", voice="af_sky")
        self.tutor.item = {
            "id": "moon",
            "stem": "At night we can see the bright",
            "expected": "moon",
            "full": "At night we can see the bright moon.",
        }
        self.tutor.handle_transcript("Moon")
        line = self.tutor._pending_line
        self.assertIn("Great job, Sam", line)
        self.assertRegex(line.lower(), r"indeed moon")
        self.assertNotIn("Great job, Moon", line)
        self.assertEqual(self.tutor.voice_id, "af_sky")
        self.assertEqual(self.tutor.state.best, 1)

    def test_turns_count_misses_and_keep_best_streak(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        self.tutor.handle_transcript("the food bowl")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        self.tutor.handle_transcript("the food bowl")
        self.tutor.handle_transcript("spaceship lasagna")
        self.assertEqual(self.tutor.state.turns, 3)
        self.assertEqual(self.tutor.state.streak, 0)
        self.assertEqual(self.tutor.state.best, 2)

    def test_i_dont_know_nudges_then_reveals(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        first = self.tutor.current_item()["id"]
        self.tutor.handle_transcript("I don't know")
        self.assertEqual(self.tutor.current_item()["id"], first)
        self.assertNotIn("food bowl", self.tutor._pending_line.lower())
        self.tutor.handle_transcript("I dunno")
        self.assertEqual(self.tutor.current_item()["id"], first)
        self.tutor.handle_transcript("no idea")
        self.assertIn("food bowl", self.tutor._pending_line.lower())
        self.assertNotEqual(self.tutor.current_item()["id"], first)

    def test_idle_nudges_then_reveals(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        first = self.tutor.current_item()["id"]
        self.tutor.mark_ready()
        self.tutor.handle_idle()
        self.tutor.mark_ready()
        self.tutor.handle_idle()
        self.assertEqual(self.tutor.current_item()["id"], first)
        self.tutor.mark_ready()
        self.tutor.handle_idle()
        self.assertIn("food bowl", self.tutor._pending_line.lower())
        self.assertNotEqual(self.tutor.current_item()["id"], first)

    def test_similarity_bounds(self):
        self.assertEqual(similarity("a", "a"), 1.0)
        self.assertEqual(similarity("", "x"), 0.0)

    def test_story_accepts_synonym_and_rewards(self):
        llm = ScriptedLLM('{"correct": true}')
        tutor = LanguageTutor(llm=llm, stt=None, tts=None)
        tutor.set_mode("story")
        self.assertEqual(tutor.state.mode, "story")
        ui = None
        for i in range(5):
            tutor.item = {
                "id": f"syn-{i}",
                "sentence": "The happy puppy ran to the park.",
                "target": "happy",
                "speak": "The happy puppy ran to the park. What's another word for happy?",
                "stem": "The happy puppy ran to the park.",
                "expected": "happy",
                "full": "The happy puppy ran to the park.",
            }
            events = tutor.handle_transcript("glad")
            ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
            self.assertTrue(ui.payload["correct"])
        self.assertTrue(ui.payload.get("reward"))
        self.assertEqual(tutor.state.streak, 5)
        self.assertNotIn("whole story", tutor._pending_line.lower())
        self.assertIn("another word", tutor._pending_line.lower())

    def test_guard_rejects_slang(self):
        self.tutor.set_mode("complete")
        self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
        events = self.tutor.handle_transcript("stupid crap")
        ui = next(e for e in events if e.type == "ui" and "correct" in e.payload)
        self.assertFalse(ui.payload["correct"])
        self.assertEqual(self.tutor.state.streak, 0)

    def test_reward_after_five_sentence_wins(self):
        self.tutor.set_mode("complete")
        for i in range(5):
            self.tutor.item = copy.deepcopy(COMPLETION_ITEMS[0])
            events = self.tutor.handle_transcript("the food bowl")
        ui = next(e for e in events if e.type == "ui" and e.payload.get("correct") is True)
        self.assertTrue(ui.payload.get("reward"))
        self.assertEqual(self.tutor.state.streak, 5)


if __name__ == "__main__":
    unittest.main()
