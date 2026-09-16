"""Language tutor brain: two kid modes, in-memory audio, injectable engines."""

from __future__ import annotations

import copy
import io
import json
import os
import random
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Curriculum (used for tests, cold-start, and vLLM fallback)
# ---------------------------------------------------------------------------

COMPLETION_ITEMS = [
    {
        "id": "puppy",
        "stem": "The hungry puppy ran toward",
        "expected": "the food bowl",
        "full": "The hungry puppy ran toward the food bowl.",
    },
    {
        "id": "moon",
        "stem": "At night we can see the bright",
        "expected": "moon",
        "full": "At night we can see the bright moon.",
    },
    {
        "id": "park",
        "stem": "After school we like to play at the",
        "expected": "park",
        "full": "After school we like to play at the park.",
    },
    {
        "id": "story",
        "stem": "Grandma opened the book and began to",
        "expected": "read a story",
        "full": "Grandma opened the book and began to read a story.",
    },
    {
        "id": "rain",
        "stem": "We jumped in puddles because it was",
        "expected": "raining",
        "full": "We jumped in puddles because it was raining.",
    },
]

MISTAKE_ITEMS = [
    {
        "id": "dont",
        "kind": "grammar",
        "spoken": "She don't like apples.",
        "correct": "She doesn't like apples.",
        "hint": "Listen to the helper verb.",
    },
    {
        "id": "goed",
        "kind": "grammar",
        "spoken": "Yesterday I goed to the park.",
        "correct": "Yesterday I went to the park.",
        "hint": "The past tense of go is a special word.",
    },
    {
        "id": "have",
        "kind": "grammar",
        "spoken": "He have two cats.",
        "correct": "He has two cats.",
        "hint": "He / she / it uses a different have.",
    },
    {
        "id": "see-sea",
        "kind": "pronunciation",
        "spoken": "The ship is sailing on the see.",
        "correct": "The ship is sailing on the sea.",
        "hint": "See and sea sound alike — which spelling belongs in the ocean?",
    },
    {
        "id": "three",
        "kind": "pronunciation",
        "spoken": "I have free cookies.",
        "correct": "I have three cookies.",
        "hint": "Put your tongue between your teeth for th.",
    },
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [t for t in normalize_text(text).split(" ") if t]


def pseudo_phonemes(text: str) -> str:
    """Tiny grapheme map so tests can score pronunciation without a G2P model."""
    s = normalize_text(text)
    replacements = (
        ("tion", "shun"),
        ("ough", "o"),
        ("ph", "f"),
        ("wh", "w"),
        ("kn", "n"),
        ("wr", "r"),
        ("ck", "k"),
        ("th", "T"),
        ("ch", "C"),
        ("sh", "S"),
        ("ee", "i"),
        ("ea", "i"),
        ("oo", "u"),
        ("qu", "kw"),
    )
    for a, b in replacements:
        s = s.replace(a, b)
    s = re.sub(r"[aeiou]+", "a", s)
    s = re.sub(r"\s+", "", s)
    return s


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def phoneme_score(hypothesis: str, reference: str) -> float:
    return SequenceMatcher(None, pseudo_phonemes(hypothesis), pseudo_phonemes(reference)).ratio()


def contains_expected(hypothesis: str, expected: str, threshold: float = 0.72) -> bool:
    hyp = normalize_text(hypothesis)
    exp = normalize_text(expected)
    if not exp:
        return False
    if exp in hyp:
        return True
    # allow dropping tiny words
    exp_tokens = [t for t in tokenize(expected) if t not in {"the", "a", "an", "to", "of"}]
    if exp_tokens and all(t in hyp.split() or t in hyp for t in exp_tokens):
        return True
    return similarity(hyp, exp) >= threshold


def parse_llm_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"speak": raw, "feedback": raw}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {"speak": raw}
    except json.JSONDecodeError:
        return {"speak": raw, "feedback": raw}


@dataclass
class TutorEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    audio: Optional[bytes] = None


@dataclass
class SessionState:
    mode: str = "complete"  # complete | mistake
    phase: str = "idle"  # idle | prompt | listening | evaluating
    awaiting_completion: bool = False
    streak: int = 0
    turns: int = 0


class LanguageModel:
    """OpenAI-compatible client (vLLM). Tests inject a fake."""

    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: float = 8.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def complete(self, system: str, user: str, max_tokens: int = 220) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to talk to vLLM") from exc

        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class SpeechToText:
    def __init__(self, model_size: str = "base.en", device: str = "cuda", compute_type: str = "float16"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.device
            compute = self.compute_type
            try:
                self._model = WhisperModel(self.model_size, device=device, compute_type=compute)
            except Exception:
                self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self._model

    def transcribe(self, pcm: np.ndarray, sample_rate: int = 16000) -> str:
        if pcm is None or len(pcm) == 0:
            return ""
        model = self._load()
        audio = np.asarray(pcm, dtype=np.float32)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class TextToSpeech:
    def __init__(self, model_path: str, voices_path: str, voice: str = "af_sarah", speed: float = 1.05, lang: str = "en-us"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._kokoro = None

    def _load(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(self.model_path, self.voices_path)
        return self._kokoro

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32), 24000
        kokoro = self._load()
        samples, sample_rate = kokoro.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
        return np.asarray(samples, dtype=np.float32), int(sample_rate)


def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def decode_webm_opus(data: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Decode concatenated MediaRecorder WebM/Opus chunks in-memory via ffmpeg pipes."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError((proc.stderr or b"ffmpeg failed").decode("utf-8", errors="ignore")[:300])
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def int16_bytes_to_pcm(data: bytes) -> np.ndarray:
    if not data or len(data) < 2:
        return np.zeros(0, dtype=np.float32)
    usable = data[: len(data) - (len(data) % 2)]
    return np.frombuffer(usable, dtype=np.int16).astype(np.float32) / 32768.0


def wav_bytes_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


class ChunkAssembler:
    def __init__(self, sample_rate: int = 16000, silence_rms: float = 0.012, silence_ms: int = 450, min_speech_ms: int = 400):
        self.sample_rate = sample_rate
        self.silence_rms = silence_rms
        self.silence_samples = int(sample_rate * silence_ms / 1000)
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.reset()

    def clone(self) -> "ChunkAssembler":
        return ChunkAssembler(
            sample_rate=self.sample_rate,
            silence_rms=self.silence_rms,
            silence_ms=int(1000 * self.silence_samples / max(self.sample_rate, 1)),
            min_speech_ms=int(1000 * self.min_speech_samples / max(self.sample_rate, 1)),
        )

    def reset(self) -> None:
        self._speech = np.zeros(0, dtype=np.float32)
        self._trail = 0
        self._heard = False

    def push(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        if pcm is None or len(pcm) == 0:
            return None
        pcm = np.asarray(pcm, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(pcm)) + 1e-12))
        if rms >= self.silence_rms:
            self._heard = True
            self._trail = 0
            self._speech = np.concatenate([self._speech, pcm])
            return None
        if not self._heard:
            return None
        self._speech = np.concatenate([self._speech, pcm])
        self._trail += len(pcm)
        if self._trail >= self.silence_samples and len(self._speech) >= self.min_speech_samples:
            utterance = self._speech.copy()
            self.reset()
            return utterance
        return None


SYSTEM_COMPLETE = textwrap.dedent("""\
    You are Teddy, a warm teddy-bear language tutor for children ages 5-8.
    Speak in short, cheerful sentences. Never use sarcasm or scolding.
    YOU are the judge: decide if the child's words reasonably complete the stem.
    Invent a BRAND NEW next stem (different topic from recent ones).
    Reply with JSON only:
    {"correct": true/false, "feedback": "short UI text",
     "speak": "one spoken line: praise or hint, THEN the next stem ending with ...",
     "next": {"id": "short-id", "stem": "new first half", "expected": "likely ending", "full": "full sentence"}}
""")

SYSTEM_MISTAKE = textwrap.dedent("""\
    You are Teddy, a warm teddy-bear language tutor for children ages 5-8.
    YOU are the judge: did the child say the corrected sentence (or a fair kid phrasing of it)?
    Invent a BRAND NEW next mistake (alternate grammar vs pronunciation).
    Reply with JSON only:
    {"correct": true/false, "feedback": "short UI text", "issue": "grammar|pronunciation|none",
     "speak": "one spoken line: praise or hint, THEN say the next broken sentence and ask what it should be",
     "next": {"id": "short-id", "kind": "grammar|pronunciation", "spoken": "broken sentence",
              "correct": "fixed sentence", "hint": "kid hint"}}
""")

SYSTEM_OPENER = textwrap.dedent("""\
    You are Teddy, inventing one new kid language prompt. JSON only.
    If mode is complete: {"id":"...","stem":"...","expected":"...","full":"..."}
    If mode is mistake: {"id":"...","kind":"grammar|pronunciation","spoken":"...","correct":"...","hint":"..."}
    Use simple words. Do not repeat recent_ids.
""")


class LanguageTutor:
    def __init__(
        self,
        llm: Optional[LanguageModel] = None,
        stt: Optional[SpeechToText] = None,
        tts: Optional[TextToSpeech] = None,
        sample_rate: int = 16000,
        assembler: Optional[ChunkAssembler] = None,
    ):
        self.llm = llm
        self.stt = stt
        self.tts = tts
        self.sample_rate = sample_rate
        self.assembler = assembler or ChunkAssembler(sample_rate=sample_rate)
        self.state = SessionState()
        self.webm_buf = bytearray()
        self.item: dict[str, Any] = copy.deepcopy(COMPLETION_ITEMS[0])
        self.recent: list[str] = []
        self.busy = False
        self.accepting = False
        self._pending_line = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LanguageTutor":
        llm_cfg = config.get("llm", {})
        stt_cfg = config.get("stt", {})
        tts_cfg = config.get("tts", {})
        audio_cfg = config.get("audio", {})
        model_dir = Path(tts_cfg.get("model_dir", "audio_utils/tts/kokoro-tts/models"))
        if not model_dir.is_absolute():
            model_dir = Path(__file__).resolve().parent / model_dir
        llm = LanguageModel(
            base_url=llm_cfg.get("base_url", "http://127.0.0.1:8000/v1"),
            model=llm_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
            api_key=llm_cfg.get("api_key", "local"),
            timeout=float(llm_cfg.get("timeout", 12)),
        )
        stt = SpeechToText(
            model_size=stt_cfg.get("model", "tiny.en"),
            device=stt_cfg.get("device", "cpu"),
            compute_type=stt_cfg.get("compute_type", "int8"),
        )
        tts = TextToSpeech(
            model_path=str(model_dir / tts_cfg.get("onnx", "kokoro-v1.0.onnx")),
            voices_path=str(model_dir / tts_cfg.get("voices", "voices-v1.0.bin")),
            voice=tts_cfg.get("voice", "af_sarah"),
            speed=float(tts_cfg.get("speed", 1.05)),
            lang=tts_cfg.get("lang", "en-us"),
        )
        assembler = ChunkAssembler(
            sample_rate=int(audio_cfg.get("sample_rate", 16000)),
            silence_rms=float(audio_cfg.get("silence_rms", 0.012)),
            silence_ms=int(audio_cfg.get("silence_ms", 450)),
            min_speech_ms=int(audio_cfg.get("min_speech_ms", 400)),
        )
        return cls(llm=llm, stt=stt, tts=tts, sample_rate=int(audio_cfg.get("sample_rate", 16000)), assembler=assembler)

    def current_item(self) -> dict[str, Any]:
        return self.item

    def _bank(self) -> list[dict[str, Any]]:
        return COMPLETION_ITEMS if self.state.mode == "complete" else MISTAKE_ITEMS

    def _valid_item(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if self.state.mode == "complete":
            return bool(item.get("stem") and item.get("expected"))
        return bool(item.get("spoken") and item.get("correct") and item.get("kind") in {"grammar", "pronunciation", None, "Grammar", "Pronunciation"})

    def _remember(self, item: dict[str, Any]) -> None:
        key = str(item.get("id") or item.get("stem") or item.get("spoken") or "")
        if key:
            self.recent = (self.recent + [key])[-8:]

    def pick_random_item(self, exclude_current: bool = True) -> dict[str, Any]:
        bank = self._bank()
        current_id = self.item.get("id") if exclude_current else None
        options = [copy.deepcopy(x) for x in bank if x.get("id") != current_id and x.get("id") not in self.recent]
        if not options:
            options = [copy.deepcopy(x) for x in bank if x.get("id") != current_id]
        chosen = copy.deepcopy(random.choice(options or bank))
        chosen.setdefault("id", chosen.get("stem") or chosen.get("spoken") or "item")
        if chosen.get("kind"):
            chosen["kind"] = str(chosen["kind"]).lower()
        if self.state.mode == "complete":
            chosen.setdefault("full", f"{chosen['stem']} {chosen['expected']}.")
        self.item = chosen
        self._remember(chosen)
        return chosen

    def _advance(self) -> None:
        self.state.turns += 1
        self.pick_random_item(exclude_current=True)

    def set_mode(self, mode: str) -> None:
        self.state = SessionState(mode="mistake" if mode == "mistake" else "complete")
        self.assembler.reset()
        self.webm_buf.clear()
        self.busy = False
        self.accepting = False
        self.pick_random_item(exclude_current=False)

    def prompt_speech(self) -> str:
        item = self.current_item()
        if self.state.mode == "complete":
            return f"{item['stem']} ... can you finish that sentence?"
        kind = "silly grammar" if item.get("kind") == "grammar" else "tricky sound"
        return f"Listen for the {kind} mistake. {item['spoken']} What should it be?"

    def ui_snapshot(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        item = self.current_item()
        data = {
            "mode": self.state.mode,
            "phase": self.state.phase,
            "streak": self.state.streak,
            "turns": self.state.turns,
            "item_id": item.get("id"),
        }
        if self.state.mode == "complete":
            data.update({"stem": item.get("stem", ""), "prompt": f"{item.get('stem', '')} …"})
        else:
            data.update(
                {
                    "broken": item.get("spoken", ""),
                    "kind": item.get("kind", "grammar"),
                    "hint": item.get("hint", ""),
                    "prompt": item.get("spoken", ""),
                }
            )
        if extra:
            data.update(extra)
        return data

    def speak_bytes(self, text: str) -> Optional[bytes]:
        if not text or self.tts is None:
            return None
        try:
            samples, sr = self.tts.synthesize(text)
            if samples is None or len(samples) == 0:
                return None
            return pcm_to_wav_bytes(samples, sr)
        except Exception:
            return None

    def _events_for_speech(self, speak: str, ui: dict[str, Any], state: str, with_audio: bool = True) -> list[TutorEvent]:
        events = [
            TutorEvent("state", {"state": state, **ui}),
            TutorEvent("ui", ui),
        ]
        if speak:
            events.append(TutorEvent("speak_text", {"text": speak}))
        if with_audio:
            audio = self.speak_bytes(speak) if speak else None
            if audio:
                events.append(TutorEvent("audio", {"mime": "audio/wav"}, audio=audio))
        return events

    def start_turn(self) -> list[TutorEvent]:
        self.assembler.reset()
        self.webm_buf.clear()
        self.busy = False
        self.accepting = False
        if not self.item:
            self.pick_random_item(exclude_current=False)
        self.state.phase = "prompt"
        self.state.awaiting_completion = True
        speak = self.prompt_speech()
        ui = self.ui_snapshot({"feedback": "Teddy is talking. Get ready to answer."})
        events = self._events_for_speech(speak, ui, "speaking", with_audio=False)
        self.state.phase = "listening"
        self._pending_line = speak
        return events

    def pending_speech_audio(self) -> Optional[bytes]:
        line = getattr(self, "_pending_line", "") or self.prompt_speech()
        return self.speak_bytes(line)

    def mark_ready(self) -> None:
        self.accepting = True
        self.busy = False
        self.assembler.reset()
        self.state.phase = "listening"

    def evaluate_completion(self, transcript: str) -> dict[str, Any]:
        item = self.current_item()
        expected = item.get("expected", "")
        full = item.get("full") or f"{item.get('stem', '')} {expected}"
        ok = contains_expected(transcript, expected) or contains_expected(transcript, full)
        word_score = similarity(transcript, expected)
        phone = phoneme_score(transcript, expected)
        if ok:
            speak = f"Yes! {full} Super job finishing that thought!"
            feedback = "You completed the sentence."
        elif phone >= 0.82 and not ok:
            speak = f"So close! I think you meant {expected}. Let's say it clearly: {full}"
            feedback = "Almost — let's polish the ending."
        else:
            speak = f"Nice try! The sentence can end with {expected}. Say it with me next time."
            feedback = "Not quite yet — listen to Teddy's ending."
        return {
            "correct": ok,
            "speak": speak,
            "feedback": feedback,
            "word_score": round(word_score, 3),
            "phoneme_score": round(phone, 3),
        }

    def evaluate_mistake(self, transcript: str) -> dict[str, Any]:
        item = self.current_item()
        said = normalize_text(transcript)
        sim_broken = similarity(said, item.get("spoken", ""))
        sim_fix = similarity(said, item.get("correct", ""))
        phone = phoneme_score(transcript, item.get("correct", ""))
        repeated_error = sim_broken >= sim_fix
        said_fix = contains_expected(transcript, item.get("correct", ""), threshold=0.86) or sim_fix >= 0.9
        if item.get("kind") == "pronunciation":
            said_fix = said_fix or phone >= 0.88
        ok = bool(said_fix and not repeated_error)
        if ok:
            speak = f"You found it! The right sentence is: {item['correct']} I'm proud of you."
            feedback = "Great ears — that's the fix."
            issue = "none"
        elif repeated_error:
            speak = f"That was the tricky version. The fix is: {item['correct']}. {item.get('hint', '')}"
            feedback = "You repeated the mistake — here's the repair."
            issue = item.get("kind", "grammar")
        elif phone >= 0.8:
            speak = f"I heard you almost say it. Stretch the sounds: {item['correct']}"
            feedback = "Meaning is close; pronunciation needs a tiny tweak."
            issue = "pronunciation"
        else:
            speak = f"Let's hunt together. {item.get('hint', '')} Try: {item['correct']}"
            feedback = "Keep going — Teddy will help."
            issue = item.get("kind", "grammar")
        return {
            "correct": ok,
            "speak": speak,
            "feedback": feedback,
            "issue": issue,
            "word_score": round(similarity(transcript, item.get("correct", "")), 3),
            "phoneme_score": round(phone, 3),
        }

    def _apply_next(self, parsed: dict[str, Any], correct: bool) -> None:
        nxt = parsed.get("next")
        if correct and self._valid_item(nxt):
            nxt["kind"] = str(nxt.get("kind", "grammar")).lower()
            nxt.setdefault("id", nxt.get("stem") or nxt.get("spoken"))
            self.item = nxt
            self._remember(nxt)
            self.state.turns += 1
            return
        if correct:
            self._advance()

    def _maybe_llm(self, transcript: str, rule: dict[str, Any]) -> dict[str, Any]:
        if self.llm is None:
            return rule
        item = self.current_item()
        try:
            payload = {"item": item, "child": transcript, "rule_hint": {"correct": rule["correct"]}, "recent_ids": self.recent}
            system = SYSTEM_COMPLETE if self.state.mode == "complete" else SYSTEM_MISTAKE
            raw = self.llm.complete(system, json.dumps(payload))
            parsed = parse_llm_json(raw)
            if not parsed:
                return rule
            merged = dict(rule)
            for key in ("speak", "feedback", "correct", "issue", "next"):
                if key in parsed:
                    merged[key] = parsed[key]
            merged["correct"] = bool(merged.get("correct"))
            speak = str(merged.get("speak") or rule["speak"])
            if len(speak) > 320:
                speak = speak[:317] + "..."
            merged["speak"] = speak
            return merged
        except Exception:
            return rule

    def handle_transcript(self, transcript: str) -> list[TutorEvent]:
        transcript = (transcript or "").strip()
        events = [TutorEvent("transcript", {"text": transcript, "partial": False, "heard": transcript})]
        if not transcript:
            events.append(
                TutorEvent(
                    "state",
                    {"state": "listening", **self.ui_snapshot({"feedback": "I didn't catch that. Try again a little louder.", "heard": ""})},
                )
            )
            return events
        self.state.phase = "evaluating"
        events.append(TutorEvent("state", {"state": "thinking", "feedback": "Teddy is thinking…", "heard": transcript, **self.ui_snapshot()}))
        rule = self.evaluate_completion(transcript) if self.state.mode == "complete" else self.evaluate_mistake(transcript)
        result = self._maybe_llm(transcript, rule)
        if result["correct"]:
            self.state.streak += 1
        else:
            self.state.streak = 0
        self._apply_next(result, result["correct"])
        speak = result["speak"]
        follow = self.prompt_speech()
        if follow.lower() not in speak.lower():
            speak = f"{speak} {follow}"
        ui = self.ui_snapshot(
            {
                "feedback": result["feedback"],
                "correct": result["correct"],
                "word_score": result.get("word_score"),
                "phoneme_score": result.get("phoneme_score"),
                "issue": result.get("issue"),
                "heard": transcript,
            }
        )
        events.extend(self._events_for_speech(speak, ui, "speaking", with_audio=False))
        self._pending_line = speak
        self.state.phase = "listening"
        return events

    def transcribe_pcm(self, pcm: np.ndarray) -> str:
        if self.stt is None:
            return ""
        return self.stt.transcribe(pcm, self.sample_rate)

    def take_utterance(self, pcm: np.ndarray, force: bool = False) -> Optional[np.ndarray]:
        if self.busy or not self.accepting:
            return None
        if force and pcm is not None and len(pcm) > 0:
            utterance = np.asarray(pcm, dtype=np.float32)
            self.assembler.reset()
        else:
            utterance = self.assembler.push(pcm)
        if utterance is None:
            return None
        self.busy = True
        self.accepting = False
        return utterance

    def finish_utterance(self, pcm: np.ndarray) -> list[TutorEvent]:
        try:
            return self.handle_transcript(self.transcribe_pcm(pcm))
        finally:
            self.busy = False

    def ingest_pcm(self, pcm: np.ndarray, force: bool = False) -> list[TutorEvent]:
        utterance = self.take_utterance(pcm, force=force)
        if utterance is None:
            return []
        return self.finish_utterance(utterance)

    def ingest_webm(self, chunk: bytes, utterance_end: bool = False) -> list[TutorEvent]:
        if chunk:
            self.webm_buf.extend(chunk)
        if not utterance_end:
            return []
        blob = bytes(self.webm_buf)
        self.webm_buf.clear()
        self.assembler.reset()
        if not blob:
            return []
        pcm = decode_webm_opus(blob, self.sample_rate)
        return self.ingest_pcm(pcm, force=True)
