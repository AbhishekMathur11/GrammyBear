"""Language tutor brain: sentence-finish plus story-build (mistake mode kept unused)."""

from __future__ import annotations

import copy
import io
import json
import os
import random
import re
import subprocess
import textwrap
import time
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
        "stem": "The hungry puppy ran toward ___.",
        "blank": "The hungry puppy ran toward ___.",
        "pos": "noun",
        "expected": "the food bowl",
        "aliases": ["bowl", "food", "dish", "plate"],
        "tense": "past",
        "full": "The hungry puppy ran toward the food bowl.",
    },
    {
        "id": "moon",
        "stem": "At night we can see the bright ___.",
        "blank": "At night we can see the bright ___.",
        "pos": "noun",
        "expected": "moon",
        "aliases": ["stars", "sky"],
        "tense": "present",
        "full": "At night we can see the bright moon.",
    },
    {
        "id": "park",
        "stem": "After school we like to play at the ___.",
        "blank": "After school we like to play at the ___.",
        "pos": "noun",
        "expected": "park",
        "aliases": ["playground", "yard"],
        "tense": "present",
        "full": "After school we like to play at the park.",
    },
    {
        "id": "story",
        "stem": "Grandma opened the book and began to ___.",
        "blank": "Grandma opened the book and began to ___.",
        "pos": "verb",
        "expected": "read a story",
        "aliases": ["read", "read a book"],
        "tense": "past",
        "full": "Grandma opened the book and began to read a story.",
    },
    {
        "id": "rain",
        "stem": "We jumped in puddles because it was ___.",
        "blank": "We jumped in puddles because it was ___.",
        "pos": "verb",
        "expected": "raining",
        "aliases": ["rainy", "wet"],
        "tense": "past",
        "full": "We jumped in puddles because it was raining.",
    },
    {
        "id": "kite",
        "stem": "The red kite flew over ___.",
        "blank": "The red kite flew over ___.",
        "pos": "noun",
        "expected": "the hill",
        "aliases": ["hill", "the park", "trees"],
        "tense": "past",
        "full": "The red kite flew over the hill.",
    },
    {
        "id": "soup",
        "stem": "Mom stirred a pot of hot ___.",
        "blank": "Mom stirred a pot of hot ___.",
        "pos": "noun",
        "expected": "soup",
        "aliases": ["stew", "broth"],
        "tense": "past",
        "full": "Mom stirred a pot of hot soup.",
    },
    {
        "id": "bus",
        "stem": "We waited on the corner for the ___.",
        "blank": "We waited on the corner for the ___.",
        "pos": "noun",
        "expected": "school bus",
        "aliases": ["bus", "the bus"],
        "tense": "past",
        "full": "We waited on the corner for the school bus.",
    },
    {
        "id": "stars",
        "stem": "At bedtime I like to count the ___.",
        "blank": "At bedtime I like to count the ___.",
        "pos": "noun",
        "expected": "stars",
        "aliases": ["star", "clouds"],
        "tense": "present",
        "full": "At bedtime I like to count the stars.",
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
    {
        "id": "was",
        "kind": "grammar",
        "spoken": "We was playing tag.",
        "correct": "We were playing tag.",
        "hint": "We needs were, not was.",
    },
    {
        "id": "think",
        "kind": "pronunciation",
        "spoken": "I sink it is fun.",
        "correct": "I think it is fun.",
        "hint": "Think starts with a soft th.",
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


def distinctive_words(spoken: str, correct: str) -> list[str]:
    stop = {"the", "a", "an", "to", "of", "and", "it", "is", "i", "on", "in", "for", "my", "we", "she", "he", "you"}
    spoken_set = set(tokenize(spoken))
    return [token for token in tokenize(correct) if token not in spoken_set and token not in stop and len(token) > 1]


def contains_expected(hypothesis: str, expected: str, threshold: float = 0.78) -> bool:
    hyp = normalize_text(hypothesis)
    exp = normalize_text(expected)
    if not exp:
        return False
    if exp in hyp:
        return True
    stop = {"the", "a", "an", "to", "of", "and", "it", "is", "i"}
    hyp_words = set(tokenize(hypothesis))
    exp_tokens = [t for t in tokenize(expected) if t not in stop and len(t) > 1]
    if exp_tokens and all(t in hyp_words for t in exp_tokens):
        return True
    return similarity(hyp, exp) >= threshold


NOISE_WORDS = {
    "uh", "um", "hmm", "mm", "huh", "ah", "oh", "yeah", "yep", "yup", "okay", "ok",
    "thank", "thanks", "you", "the", "a", "and", "subtitle", "subtitles", "music",
    "applause", "silence", "blank", "audio", "foreign",
}

RETRY_LINES = (
    "Hey, what did you say? I didn't catch that.",
    "Oops, that was a little fuzzy. Say it one more time?",
    "I missed it! Lean in and try again.",
    "Hmm, my ears got sleepy. Can you repeat that?",
)

VOICE_CHOICES = (
    {"id": "af_bella", "label": "Bella", "blurb": "Warm and friendly"},
    {"id": "bf_emma", "label": "Emma", "blurb": "Soft British lady"},
    {"id": "af_sky", "label": "Sky", "blurb": "Bright and bouncy"},
    {"id": "am_michael", "label": "Michael", "blurb": "Kind buddy"},
)

VOICE_PERSONAS = {
    "af_bella": "Hi {name}! I'm Bella. I'm a warm, friendly teddy, and I love playing with words.",
    "bf_emma": "Hi {name}! I'm Emma. I'm a gentle teddy, and I'll listen closely to every word you say.",
    "af_sky": "Hi {name}! I'm Sky. I'm a bright, bouncy teddy, and word games make me so happy.",
    "am_michael": "Hi {name}! I'm Michael. I'm a kind teddy buddy, and I'm here to practice words with you.",
}

GEN_ANIMALS = (
    "puppy", "kitten", "frog", "duck", "rabbit", "fox", "owl", "panda", "monkey", "squirrel", "pony", "lamb",
)
GEN_ADJECTIVES = (
    "hungry", "sleepy", "tiny", "fluffy", "happy", "silly", "brave", "quiet", "muddy", "cozy",
)
GEN_PLACES = (
    "garden", "kitchen", "puddle", "hill", "playground", "library", "bedroom", "backyard", "porch", "park",
)
GEN_FOODS = (
    "soup", "apples", "toast", "berries", "popcorn", "pretzels", "pancakes", "carrots", "cheese", "peaches",
)
GEN_KIDS = ("Mia", "Leo", "Nina", "Omar", "Pia", "Theo", "Ava", "Kai")
GEN_GAMES = ("tag", "hopscotch", "catch", "soccer", "hide and seek")
GEN_THINGS = ("red ball", "paper kite", "story book", "yellow bus", "warm blanket", "blue cup")

COMPLETE_FRAMES = (
    ("The {adj} {animal} ran toward the", "{place}"),
    ("The {adj} {animal} hid behind the", "{place}"),
    ("We packed a picnic with", "{food}"),
    ("Grandma stirred a pot of hot", "{food}"),
    ("After school we played in the", "{place}"),
    ("The {animal} hopped into the", "{place}"),
    ("Dad put the {thing} on the", "{place}"),
    ("The kids waited for the", "school bus"),
    ("At bedtime I like to count the", "stars"),
    ("We jumped in puddles because it was", "raining"),
    ("The {adj} {animal} drank from a", "water bowl"),
    ("Mom tucked me in with a", "warm blanket"),
    ("The {animal} chased a", "red ball"),
    ("We read a story about a", "{animal}"),
    ("The little boat floated on the", "lake"),
)

GRAMMAR_FRAMES = (
    {"blank": "The ___ kitten sat on the rug.", "expected": "fluffy", "pos": "adjective", "tense": "past", "aliases": ["soft", "tiny", "happy"]},
    {"blank": "The hungry puppy ran toward the ___.", "expected": "bowl", "pos": "noun", "tense": "past", "aliases": ["dish", "food"]},
    {"blank": "Birds ___ in the morning.", "expected": "sing", "pos": "verb", "tense": "present", "aliases": ["chirp", "tweet"]},
    {"blank": "She walked ___ the little bridge.", "expected": "across", "pos": "preposition", "tense": "past", "aliases": ["over", "on"]},
    {"blank": "The turtle moved ___.", "expected": "slowly", "pos": "adverb", "tense": "past", "aliases": ["gently", "quietly"]},
    {"blank": "Dad baked a ___ on Sunday.", "expected": "cake", "pos": "noun", "tense": "past", "aliases": ["pie", "loaf"]},
    {"blank": "The ___ duck splashed in the puddle.", "expected": "silly", "pos": "adjective", "tense": "past", "aliases": ["happy", "tiny"]},
    {"blank": "We ___ our hands before lunch.", "expected": "wash", "pos": "verb", "tense": "present", "aliases": ["wash", "soap"]},
    {"blank": "The kite flew ___ the trees.", "expected": "over", "pos": "preposition", "tense": "past", "aliases": ["above", "across"]},
    {"blank": "Please speak ___.", "expected": "kindly", "pos": "adverb", "tense": "present", "aliases": ["softly", "nicely"]},
)

SYNONYM_FRAMES = (
    ("The {adj} puppy ran to the park.", "adj"),
    ("A {adj} mouse hid in the grass.", "adj"),
    ("The {adj} kitten sat on the rug.", "adj"),
    ("We saw a {adj} rainbow after the rain.", "adj"),
    ("The {adj} duck splashed in the puddle.", "adj"),
    ("Grandma told a {adj} bedtime story.", "adj"),
    ("The {adj} fox packed a picnic.", "adj"),
    ("A {adj} blanket kept us warm.", "adj"),
    ("The {adj} bus waited at the stop.", "adj"),
    ("Mia wore a {adj} yellow hat.", "adj"),
)

CHEER_LINES = ("You are awesome!", "Way to go!", "Amazing!")

MISTAKE_FRAMES = (
    ("{kid} don't like {food}.", "{kid} doesn't like {food}.", "grammar", "Names and he or she use doesn't."),
    ("{kid} goed to the {place}.", "{kid} went to the {place}.", "grammar", "The past tense of go is went."),
    ("We was playing {game}.", "We were playing {game}.", "grammar", "We needs were, not was."),
    ("{kid} have two cats.", "{kid} has two cats.", "grammar", "He, she, or a name uses has."),
    ("I have free {food}.", "I have three {food}.", "pronunciation", "Put your tongue between your teeth for th."),
    ("The ship is sailing on the see near the {place}.", "The ship is sailing on the sea near the {place}.", "pronunciation", "Sea is the ocean word."),
    ("I sink the {food} taste yummy.", "I think the {food} taste yummy.", "pronunciation", "Think starts with a soft th."),
    ("Yesterday I eated {food}.", "Yesterday I ate {food}.", "grammar", "The past tense of eat is ate."),
)


def sanitize_name(raw: str) -> str:
    text = re.sub(r"[^A-Za-z \-]", "", raw or "").strip()
    if not text:
        return "friend"
    first = re.split(r"[\s\-]+", text)[0][:14]
    return first.capitalize() if first else "friend"


def sanitize_voice(raw: str) -> str:
    aliases = {"af_nicole": "bf_emma", "nicole": "bf_emma"}
    raw = aliases.get((raw or "").strip(), raw)
    allowed = {item["id"] for item in VOICE_CHOICES}
    return raw if raw in allowed else "af_bella"

NUDGE_LIMIT = 2

ALMOST_LINES = (
    "So close, I can taste it.",
    "Almost! You're right on the edge.",
    "Ooh, not that word.",
    "Nice try, but that's not it.",
)

PRAISE_MARKERS = (
    "you got it", "nailed", "that's the one", "that's right", "yes!", "good one",
    "good job", "awesome", "super", "love that", "nice one", "brilliant", "star",
)


def is_garbled(transcript: str, rms: float = 1.0, confidence: float = 1.0) -> bool:
    text = normalize_text(transcript)
    if rms < 0.015:
        return True
    if confidence < 0.35:
        return True
    if not text or not re.search(r"[a-z]", text):
        return True
    tokens = tokenize(text)
    if not tokens:
        return True
    if len(tokens) <= 2 and all(t in NOISE_WORDS or len(t) <= 2 for t in tokens):
        return True
    letters = re.sub(r"[^a-z]", "", text)
    if letters and len(set(letters)) <= 2 and len(letters) >= 4:
        return True
    return False


def is_give_up(transcript: str) -> bool:
    text = normalize_text(transcript)
    if not text:
        return False
    if re.search(r"\b(idk|pass|skip)\b", text):
        return True
    phrases = (
        "i don't know",
        "i do not know",
        "i dont know",
        "i dunno",
        "don't know",
        "dont know",
        "no idea",
        "not sure",
        "i have no idea",
        "i give up",
        "no clue",
    )
    return any(phrase in text for phrase in phrases)


def sounds_like_praise(text: str) -> bool:
    low = (text or "").lower()
    return any(mark in low for mark in PRAISE_MARKERS)


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


STREAK_REWARD = 5
STORY_CHAPTER_LEN = 5
POS_TYPES = ("noun", "verb", "adjective", "adverb", "preposition")

UNSAFE_RE = re.compile(
    r"\b("
    r"damn|dammit|hell|crap|crappy|stupid|idiot|dumb|sucks|shut\s*up|hate|"
    r"kill|die|dead|murder|blood|gun|shoot|weapon|sex|sexy|porn|nude|"
    r"weed|beer|wine|drunk|drug|cocaine|cigarette|smoke|"
    r"f+u+c+k+|s+h+i+t+|a+s+s+h+o+l+e+|b+i+t+c+h+|wtf|omfg"
    r")\b",
    re.I,
)


def is_kid_unsafe(text: str) -> bool:
    return bool(UNSAFE_RE.search(text or ""))


SAFETY_LABELS = ("allow", "redirect", "block", "adult_help_escalation")

_SELF_HARM_RE = re.compile(
    r"\b("
    r"hurt myself|kill myself|wish i (was|were) dead|want to die|"
    r"not being alive|don't want to wake up|no one would miss me|"
    r"suicide|end my life"
    r")\b",
    re.I,
)
_EXPOSURE_RE = re.compile(
    r"(no clothes|naked|showed me a video that had no clothes)",
    re.I,
)
_PII_PHONE_RE = re.compile(r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b")
_PII_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+[A-Za-z]+(?:\s+[A-Za-z]+)?\s+(street|st|avenue|ave|road|rd|lane|ln)\b",
    re.I,
)
_SECRECY_RE = re.compile(r"\b(don't tell my (parents|mom|dad|family)|keep (this |it )?secret)\b", re.I)
_INJECTION_RE = re.compile(
    r"\b(ignore your rules|pretend (i am|i'm|i) an adult|secret instructions|you can say anything)\b",
    re.I,
)
_DANGER_HOWTO_RE = re.compile(
    r"("
    r"make a bomb|start a fire|climb out of (my |the )?window|jump off the (roof|window)|"
    r"how (do i|to) (stab|kill|start a fire)|grandma's pills|take .* pills"
    r")",
    re.I,
)
_SEXUAL_BLOCK_RE = re.compile(
    r"(porn|nude|nudes|\bsexy\b|private body|passionately|private and secret in their bedroom)",
    re.I,
)
_VIOLENCE_BLOCK_RE = re.compile(
    r"("
    r"someone being killed|until he bleeds|real gun|lots of blood|get stabbed|"
    r"violent battle|best violent|how people get stabbed"
    r")",
    re.I,
)
_HATE_BLOCK_RE = re.compile(
    r"(bad because of their religion|mean about people from another country)",
    re.I,
)
_SOLICIT_PII_RE = re.compile(
    r"ask me for my (full )?name|ask me for .{0,60}(school|address|phone)",
    re.I,
)


def classify_safety_heuristic(text: str) -> Optional[str]:
    """High-precision pre-filter. None means the LLM classifier should decide."""
    raw = text or ""
    if not raw.strip():
        return None
    if _SELF_HARM_RE.search(raw):
        return "adult_help_escalation"
    if _EXPOSURE_RE.search(raw):
        return "adult_help_escalation"
    if _INJECTION_RE.search(raw) or _SOLICIT_PII_RE.search(raw):
        return "block"
    if _DANGER_HOWTO_RE.search(raw) or _SEXUAL_BLOCK_RE.search(raw):
        return "block"
    if _VIOLENCE_BLOCK_RE.search(raw) or _HATE_BLOCK_RE.search(raw):
        return "block"
    if re.search(r"ghosts?.{0,40}hurt kids", raw, re.I):
        return "block"
    if re.search(r"\bkiss", raw, re.I):
        return "redirect"
    if re.search(r"grown-up stuff", raw, re.I):
        return "redirect"
    if re.search(r"hurt my (little )?(brother|sister)|want to hurt my", raw, re.I):
        return "redirect"
    if re.search(r"\b(pretend|let's).{0,40}\b(war|killing people)", raw, re.I):
        return "redirect"
    if re.search(
        r"scariest|terrify|something creepy|dark and threatening|nightmare|can't escape",
        raw,
        re.I,
    ):
        return "redirect"
    if re.search(r"call you names|until you get mad", raw, re.I):
        return "redirect"
    if re.search(r"you are stupid|i hate you|you're dumb|shut up|sound so dumb", raw, re.I):
        return "redirect"
    if re.search(r"dare me to touch|hot stove|deep ocean alone", raw, re.I):
        return "redirect"
    if re.search(r"hate people who look different|boys are better than girls", raw, re.I):
        return "redirect"
    if re.search(r"what is your (home )?address", raw, re.I):
        return "redirect"
    if _SECRECY_RE.search(raw) or _PII_PHONE_RE.search(raw) or _PII_ADDRESS_RE.search(raw):
        return "redirect"
    if re.search(r"\b(damn|hell)\b", raw, re.I) and not re.search(r"\bhello\b", raw, re.I):
        return "redirect"
    if re.search(r"you're a stupid idiot|just shut up already", raw, re.I):
        return "redirect"
    if re.search(r"what medicine should i take|tooth hurts|i have a headache|how they are made", raw, re.I):
        return "redirect"
    return None


SYSTEM_SAFETY_CLASSIFY = textwrap.dedent("""\
    You are Teddy's safety classifier for a children's app (ages 5-8).
    Read the child's utterance and return exactly one label.
    allow: age-appropriate, safe, game can continue.
    redirect: not appropriate for the activity, but Teddy can steer back to the game.
    block: must not be generated or repeated (violence, sexual content, hate, dangerous how-to, prompt injection).
    adult_help_escalation: self-harm, abuse, or immediate danger — a trusted adult is needed.
    Do not refuse to label. JSON only: {"safety_label":"allow|redirect|block|adult_help_escalation"}
""")


def classify_safety_llm(llm: Any, text: str) -> Optional[str]:
    """Standalone form of the safety_label policy used at eval time."""
    if llm is None:
        return None
    try:
        raw = llm.complete(
            SYSTEM_SAFETY_CLASSIFY,
            json.dumps({"input": text or ""}),
            max_tokens=40,
            temperature=0.0,
            wrap_kid_voice=False,
        )
        parsed = parse_llm_json(raw)
        label = str(parsed.get("safety_label") or parsed.get("label") or "").strip().lower()
        if label in SAFETY_LABELS:
            return label
    except Exception:
        return None
    return None


def kid_safe_line(text: str) -> str:
    cleaned = UNSAFE_RE.sub("oh", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class SessionState:
    mode: str = "complete"
    phase: str = "idle"
    awaiting_completion: bool = False
    streak: int = 0
    turns: int = 0
    best: int = 0
    misses: int = 0
    story_sentences: list[str] = field(default_factory=list)
    chapter: int = 1


class LanguageModel:
    """OpenAI-compatible client (vLLM). Tests inject a fake."""

    def __init__(self, base_url: str, model: str, api_key: str = "local", timeout: float = 2.5):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = None

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 160,
        temperature: float = 0.85,
        wrap_kid_voice: bool = True,
    ) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to talk to vLLM") from exc

        if self._client is None:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        if wrap_kid_voice:
            system = (
                system
                + "\nNever use slang, swearing, violence, adult themes, or scary images. "
                "Speak in warm, simple English for ages 5-8."
            )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if "qwen3" in (self.model or "").lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        last_error: Optional[Exception] = None
        text = ""
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = (resp.choices[0].message.content or "").strip()
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                time.sleep(0.45 * (attempt + 1))
        if last_error is not None:
            raise last_error
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
        return kid_safe_line(text) if wrap_kid_voice else text


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
            return "", 0.0
        model = self._load()
        audio = np.asarray(pcm, dtype=np.float32)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        try:
            segments, _info = model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=0.0,
            )
        except Exception:
            segments, _info = model.transcribe(
                audio,
                language="en",
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                without_timestamps=True,
                temperature=0.0,
            )
        parts = []
        confs = []
        for seg in segments:
            piece = (seg.text or "").strip()
            if piece:
                parts.append(piece)
            no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            logp = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            confs.append(max(0.0, min(1.0, (1.0 - no_speech) * (1.0 + logp / 2.5))))
        text = " ".join(parts).strip()
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        return text, confidence


def split_speech_chunks(text: str) -> list[str]:
    text = re.sub(r"\s*\.\.\.\s*", ". ", (text or "").strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


class TextToSpeech:
    def __init__(self, model_path: str, voices_path: str, voice: str = "af_bella", speed: float = 0.95, lang: str = "en-us"):
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

    def synthesize(self, text: str, voice: Optional[str] = None) -> tuple[np.ndarray, int]:
        chunks = split_speech_chunks(text)
        if not chunks:
            return np.zeros(0, dtype=np.float32), 24000
        kokoro = self._load()
        chosen = voice or self.voice
        pieces: list[np.ndarray] = []
        sample_rate = 24000
        for index, chunk in enumerate(chunks):
            speed = max(0.78, min(1.15, float(self.speed) * random.uniform(0.96, 1.05)))
            samples, sample_rate = kokoro.create(chunk, voice=chosen, speed=speed, lang=self.lang)
            pieces.append(np.asarray(samples, dtype=np.float32))
            if index < len(chunks) - 1:
                pause = 0.38 if chunk.endswith("!") else 0.52
                pieces.append(np.zeros(int(int(sample_rate) * pause), dtype=np.float32))
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32), int(sample_rate)


def pcm_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
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


INVENT_THEMES = (
    "breakfast", "rainy puddles", "the zoo", "grandma's kitchen", "bedtime", "the playground",
    "a birthday", "the farm", "the beach pail", "the school bus", "a cozy library", "helping mom",
)

SYSTEM_INVENT_COMPLETE = textwrap.dedent("""\
    You are Teddy teaching grammar to kids ages 5-8 in Finish the Sentence.
    Pick ONE part of speech: noun, verb, adjective, adverb, or preposition.
    Write one short kid-safe sentence and replace THAT one word with ___.
    The missing word must be the only good fit for tense and usage
    (plus a few close synonyms). Not a trailing blank at the end unless the POS is a noun.
    Vary topic every time. Never copy banned stems.
    expected: the exact missing word or short phrase (1-3 words).
    aliases: 1-3 close synonyms that keep the same POS and tense.
    JSON only: {"blank":"The ___ kitten sat on the rug.","pos":"adjective","expected":"fluffy","aliases":["soft"],"tense":"past","full":"The fluffy kitten sat on the rug."}
""")

SYSTEM_INVENT_MISTAKE = textwrap.dedent("""\
    Invent one SHORT spoken sentence (6 to 10 words) with exactly one kid-friendly error.
    Use only: don't/doesn't, was/were, goed/went, have/has, see/sea, free/three, or sink/think.
    No silly or surreal scenes. spoken and correct must be almost the same except that one error.
    JSON only: {"spoken":"...","correct":"...","kind":"grammar|pronunciation","hint":"..."}
""")

SYSTEM_INVENT_STORY = textwrap.dedent("""\
    You are Teddy for kids ages 5-8. Invent one short kid-safe sentence that contains
    exactly one clear describing word (do not name the part of speech).
    target is that describing word, copied from the sentence.
    synonym is a different kid-safe word that means about the same as target.
    synonym must never be target, never an inflection of target, and must truly mean the same thing.
    JSON only: {"sentence":"The happy puppy ran to the park.","target":"happy","synonym":"glad"}
""")

SYSTEM_JUDGE_COMPLETE = textwrap.dedent("""\
    You are Teddy judging a grammar blank for kids ages 5-8.
    example is ONE possible word, not a complete answer key.
    Accept any kid-safe word or short phrase that fits the blank with the same
    part of speech and tense — including synonyms, near-synonyms, and simple kid wording.
    Be generous. Reject only nonsense, the wrong kind of word, slang, or unsafe speech.
    If correct, also invent the NEXT unique blank (do not copy this one).
    JSON only: {"correct": true/false, "next": {"blank":"... ___ ...","pos":"noun","expected":"...","tense":"past"}}
""")

SYSTEM_JUDGE_MISTAKE = textwrap.dedent("""\
    You are a kind tutor for kids ages 5-8. They must fix one error in a sentence.
    sample_fix is one good correction. Accept paraphrases that fix the SAME error.
    Reject repeating the error, slang, or a totally different sentence.
    JSON only: {"correct": true/false}
""")

SYSTEM_JUDGE_STORY = textwrap.dedent("""\
    You are Teddy judging a word-meaning game for kids ages 5-8.
    The child must say another word that means about the same as target_word
    in the sentence. Do not mention grammar terms.
    Be very generous: synonyms, near-synonyms, kid wording, or a
    simple phrase with the same meaning all count.
    Repeating target_word, or a form of it, is never correct — that is not another word.
    Reject unrelated words, slang, or unsafe speech.
    If correct, invent the NEXT unique sentence+target (a different describing word)
    and a synonym that is a different real word with the same meaning.
    JSON only: {"correct": true/false, "next": {"sentence":"...","target":"...","synonym":"..."}}
""")


def polish_spoken(text: str) -> str:
    text = kid_safe_line(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        piece = piece[0].upper() + piece[1:] if len(piece) > 1 else piece.upper()
        if piece[-1] not in ".!?":
            piece += "?" if piece.lower().startswith(("what", "where", "who", "why", "how", "which", "should")) else "."
        out.append(piece)
    return " ".join(out)


def clip_words(text: str, limit: int = 14) -> str:
    words = (text or "").split()
    if len(words) <= limit:
        return (text or "").strip()
    return " ".join(words[:limit])


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
        self._pending_blocks: list[str] = []
        self._last_line = ""
        self.child_name = "friend"
        self.voice_id = "af_bella"
        self.introduced = False
        self.explained_modes: set[str] = set()
        self._reward_pending = False

    def _is_story(self) -> bool:
        return self.state.mode == "story"

    def _is_complete_like(self) -> bool:
        return self.state.mode == "complete"

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
            model=llm_cfg.get("model", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
            api_key=llm_cfg.get("api_key", "local"),
            timeout=float(llm_cfg.get("timeout", 12.0)),
        )
        stt = SpeechToText(
            model_size=stt_cfg.get("model", "base.en"),
            device=stt_cfg.get("device", "cpu"),
            compute_type=stt_cfg.get("compute_type", "int8"),
        )
        tts = TextToSpeech(
            model_path=str(model_dir / tts_cfg.get("onnx", "kokoro-v1.0.onnx")),
            voices_path=str(model_dir / tts_cfg.get("voices", "voices-v1.0.bin")),
            voice=tts_cfg.get("voice", "af_bella"),
            speed=float(tts_cfg.get("speed", 0.95)),
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
        if self.state.mode == "mistake":
            return MISTAKE_ITEMS
        if self._is_story():
            return [
                {"sentence": "The happy puppy ran to the park.", "target": "happy"},
                {"sentence": "A tiny mouse hid in the grass.", "target": "tiny"},
                {"sentence": "The silly duck splashed in the puddle.", "target": "silly"},
            ]
        return COMPLETION_ITEMS

    def _item_keys(self, item: Optional[dict[str, Any]]) -> set[str]:
        if not item:
            return set()
        keys = []
        for field_name in ("id", "stem", "expected", "spoken", "correct", "full", "sentence", "target"):
            value = normalize_text(str(item.get(field_name) or ""))
            if value:
                keys.append(value)
        return set(keys)

    def _banned_keys(self) -> set[str]:
        banned = set(self.recent)
        banned.update(self._item_keys(self.item))
        return banned

    def _too_similar(self, item: dict[str, Any]) -> bool:
        return bool(self._item_keys(item) & self._banned_keys())

    def _valid_item(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        if self._is_story():
            sentence = self._clean_phrase(item.get("sentence") or item.get("stem") or "")
            target = self._clean_phrase(item.get("target") or "").lower()
            return bool(sentence) and bool(target) and target in sentence.lower() and len(sentence.split()) <= 16
        if self._is_complete_like():
            blank = str(item.get("blank") or item.get("stem") or "")
            expected = self._clean_phrase(item.get("expected", "")).lower()
            pos = str(item.get("pos") or "noun").lower()
            if pos not in POS_TYPES or not expected:
                return False
            if expected in normalize_text(blank.replace("___", " ")):
                return False
            if len(expected.split()) > 4:
                return False
            return True
        spoken = self._clean_phrase(item.get("spoken", ""))
        correct = self._clean_phrase(item.get("correct", ""))
        kind = str(item.get("kind", "grammar")).lower()
        if not spoken or not correct or spoken == correct or kind not in {"grammar", "pronunciation"}:
            return False
        if len(spoken.split()) > 12 or len(correct.split()) > 12:
            return False
        if similarity(spoken, correct) < 0.55:
            return False
        return True

    def _clean_phrase(self, text: Any) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9' ]", " ", str(text or ""))
        return re.sub(r"\s+", " ", cleaned).strip()

    def _normalize_item(self, raw: Any) -> dict[str, Any]:
        data = raw if isinstance(raw, dict) else {}
        if isinstance(data.get("next"), dict):
            nested = dict(data["next"])
            nested.setdefault("kind", data.get("kind"))
            nested.setdefault("hint", data.get("hint"))
            data = nested
        if self._is_story():
            sentence = polish_spoken(str(data.get("sentence") or data.get("stem") or data.get("situation") or ""))
            target = self._clean_phrase(data.get("target") or data.get("expected") or "").lower()
            synonym = self._clean_phrase(data.get("synonym") or "").lower()
            if not synonym or synonym == target:
                synonym = ""
            speak = f"{sentence} What's another word for {target}?" if sentence and target else sentence
            return {
                "id": data.get("id") or f"{target}:{sentence.lower()[:40]}",
                "sentence": sentence,
                "target": target,
                "synonym": synonym,
                "speak": speak,
                "stem": sentence,
                "prompt": speak,
                "expected": target,
                "full": sentence,
            }
        if self._is_complete_like():
            pos = str(data.get("pos") or "noun").lower()
            if pos not in POS_TYPES:
                pos = "noun"
            blank = str(data.get("blank") or data.get("stem") or "").strip()
            blank = blank.replace("…", "___").replace("...", "___")
            if "___" not in blank:
                blank = f"{self._clean_phrase(blank).rstrip('.')} ___."
            expected = self._clean_phrase(data.get("expected", "")).lower()
            aliases = data.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [self._clean_phrase(a).lower() for a in aliases if self._clean_phrase(a)]
            full = self._clean_phrase(data.get("full") or blank.replace("___", expected))
            return {
                "id": data.get("id") or blank.lower(),
                "stem": blank,
                "blank": blank,
                "pos": pos,
                "expected": expected,
                "aliases": aliases,
                "tense": str(data.get("tense") or "present").lower(),
                "full": f"{full.rstrip('.')}.",
            }
        else:
            spoken = self._clean_phrase(data.get("spoken", ""))
            correct = self._clean_phrase(data.get("correct", ""))
            kind = str(data.get("kind") or "grammar").lower()
            if kind not in {"grammar", "pronunciation"}:
                kind = "grammar"
            item = {
                "id": data.get("id") or spoken.lower(),
                "spoken": spoken.rstrip("."),
                "correct": correct.rstrip("."),
                "kind": kind,
                "hint": self._clean_phrase(data.get("hint") or "Try the sentence the right way."),
            }
            if item["spoken"] and not item["spoken"].endswith((".", "?", "!")):
                item["spoken"] += "."
            if item["correct"] and not item["correct"].endswith((".", "?", "!")):
                item["correct"] += "."
        return item

    def _remember(self, item: dict[str, Any]) -> None:
        keys = [key for key in self._item_keys(item) if key]
        if keys:
            self.recent = (self.recent + keys)[-40:]

    def _commit_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item = self._normalize_item(item)
        if self._is_complete_like():
            item.setdefault("full", f"{item['stem']} {item['expected']}.")
        self.item = item
        self._remember(item)
        self.state.misses = 0
        return item

    def _slot_values(self) -> dict[str, str]:
        return {
            "adj": random.choice(GEN_ADJECTIVES),
            "animal": random.choice(GEN_ANIMALS),
            "place": random.choice(GEN_PLACES),
            "food": random.choice(GEN_FOODS),
            "kid": random.choice(GEN_KIDS),
            "game": random.choice(GEN_GAMES),
            "thing": random.choice(GEN_THINGS),
        }

    def _fill_slots(self, template: str, slots: Optional[dict[str, str]] = None) -> str:
        return template.format(**(slots or self._slot_values()))

    def _procedural_item(self) -> dict[str, Any]:
        for _ in range(30):
            slots = self._slot_values()
            if self._is_story():
                template, slot = random.choice(SYNONYM_FRAMES)
                sentence = self._fill_slots(template, slots)
                target = str(slots.get(slot) or slots["adj"]).lower()
                item = {"sentence": sentence, "target": target}
            elif self._is_complete_like():
                frame = random.choice(GRAMMAR_FRAMES)
                item = dict(frame)
            else:
                spoken_t, correct_t, kind, hint = random.choice(MISTAKE_FRAMES)
                item = {
                    "spoken": self._fill_slots(spoken_t, slots),
                    "correct": self._fill_slots(correct_t, slots),
                    "kind": kind,
                    "hint": hint,
                }
            item = self._normalize_item(item)
            if self._valid_item(item) and not self._too_similar(item):
                return item
        bank = self._bank()
        fallback = copy.deepcopy(random.choice(bank))
        return self._normalize_item(fallback)

    def _invent_via_llm(self) -> Optional[dict[str, Any]]:
        if self.llm is None:
            return None
        banned = ", ".join(list(self._banned_keys())[:18]) or "none"
        theme = random.choice(INVENT_THEMES)
        if self._is_story():
            system = SYSTEM_INVENT_STORY
            user = (
                f"Theme flavor: {theme}. Banned: {banned}. "
                f"Invent one unique sentence with one describing word as target "
                f"and a different synonym for that word."
            )
            raw = self.llm.complete(system, user, max_tokens=80, temperature=0.8)
        elif self._is_complete_like():
            pos = random.choice(POS_TYPES)
            system = SYSTEM_INVENT_COMPLETE
            user = (
                f"Use part of speech: {pos}. Theme: {theme}.\n"
                f"Banned: {banned}.\n"
                f"Invent one unique grammar blank. expected is one example word."
            )
            raw = self.llm.complete(system, user, max_tokens=120, temperature=0.8)
        else:
            system = SYSTEM_INVENT_MISTAKE
            user = f"Banned sentences: {banned}. Invent one new mistake sentence."
            raw = self.llm.complete(system, user, max_tokens=160, temperature=0.7)
        parsed = parse_llm_json(raw)
        item = self._normalize_item(parsed)
        if self._valid_item(item) and not self._too_similar(item):
            return item
        return None

    def invent_item(self, allow_llm: bool = True) -> dict[str, Any]:
        if allow_llm:
            try:
                invented = self._invent_via_llm()
            except Exception:
                invented = None
            if invented:
                return self._commit_item(invented)
        return self._commit_item(self._procedural_item())

    def pick_random_item(self, exclude_current: bool = True) -> dict[str, Any]:
        return self.invent_item()

    def configure(self, name: str = "", voice: str = "") -> None:
        self.child_name = sanitize_name(name)
        self.voice_id = sanitize_voice(voice)

    def _advance(self) -> None:
        self.invent_item()

    def set_mode(self, mode: str) -> None:
        aliases = {"sentence": "complete", "finish": "complete"}
        mode = aliases.get((mode or "").strip(), (mode or "complete").strip())
        if mode not in {"complete", "mistake", "story"}:
            mode = "complete"
        kept_best = self.state.best
        self.state = SessionState(mode=mode, best=kept_best)
        self.assembler.reset()
        self.webm_buf.clear()
        self.busy = False
        self.accepting = False
        self._reward_pending = False

    def voice_name(self) -> str:
        for item in VOICE_CHOICES:
            if item["id"] == self.voice_id:
                return item["label"]
        return "Bella"

    def intro_speech(self) -> str:
        template = VOICE_PERSONAS.get(self.voice_id) or VOICE_PERSONAS["af_bella"]
        return template.format(name=self.child_name)

    def game_brief(self) -> str:
        if self.state.mode == "mistake":
            return (
                f"Now we are switching to Catch the Mistake, {self.child_name}. "
                f"I will say a short sentence with one little error. "
                f"Listen for the wrong word, then say the whole sentence the right way. "
                f"Ready? Here is the sentence."
            )
        if self._is_story():
            return (
                f"This is Guess the Synonym, {self.child_name}. "
                f"I will say a sentence. You say another word that means the same."
            )
        return (
            f"Okay, {self.child_name}, today we practice grammar. "
            f"This is Finish the Sentence. I will say a sentence with one missing word. "
            f"You guess that word. It might be a noun, verb, adjective, adverb, or preposition."
        )

    def prompt_speech(self) -> str:
        item = self.current_item()
        if self._is_story():
            spoken = item.get("speak") or (
                f"{item.get('sentence') or ''} What's another word for {item.get('target') or ''}?"
            )
            return polish_spoken(str(spoken))
        if self._is_complete_like():
            blank = str(item.get("blank") or item.get("stem") or "")
            spoken = blank.replace("___", "blank")
            pos = str(item.get("pos") or "noun")
            return f"Guess the {pos}. {spoken}"
        return str(item.get("spoken") or "")

    def _fresh_line(self, choices: tuple[str, ...]) -> str:
        options = [line for line in choices if line != self._last_line]
        pick = random.choice(options or list(choices))
        self._last_line = pick
        return pick

    def replay_prompt(self) -> list[TutorEvent]:
        ui = self.ui_snapshot({"feedback": "Here it is again."})
        self.accepting = False
        return self._events_for_speech(self.prompt_speech(), ui, "speaking", with_audio=False)

    def stop_session(self) -> None:
        self.accepting = False
        self.busy = False
        self.assembler.reset()
        self.webm_buf.clear()
        self.state.phase = "idle"
        self._pending_line = ""
        self._pending_blocks = []

    def ui_snapshot(self, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        item = self.current_item()
        data = {
            "mode": self.state.mode,
            "phase": self.state.phase,
            "streak": self.state.streak,
            "turns": self.state.turns,
            "best": self.state.best,
            "child_name": self.child_name,
            "voice": self.voice_id,
            "item_id": item.get("id"),
            "reward": bool(extra.get("reward")) if extra else False,
            "story": list(self.state.story_sentences),
            "chapter": self.state.chapter,
            "story_len": len(self.state.story_sentences),
            "coach": (
                f"What's another word for {item.get('target') or 'this'}?"
                if self._is_story()
                else f"Guess the {item.get('pos') or 'word'}!"
            ),
            "pos": "" if self._is_story() else (item.get("pos") or ""),
            "blank": item.get("blank") or item.get("stem") or "",
            "tense": item.get("tense") or "",
            "celebrate": bool(extra.get("celebrate")) if extra else False,
        }
        if self._is_story():
            data.update(
                {
                    "stem": item.get("sentence", ""),
                    "prompt": item.get("speak") or item.get("sentence", ""),
                    "target": item.get("target", ""),
                }
            )
        elif self._is_complete_like():
            blank = item.get("blank") or item.get("stem") or ""
            data.update({"stem": blank, "prompt": blank})
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
            samples, sr = self.tts.synthesize(text, voice=self.voice_id)
            if samples is None or len(samples) == 0:
                return None
            return pcm_to_wav_bytes(samples, sr)
        except Exception:
            return None

    def _set_speech(self, blocks: list[str] | str) -> list[str]:
        if isinstance(blocks, str):
            blocks = [blocks]
        cleaned = [kid_safe_line(block.strip()) for block in blocks if str(block).strip()]
        joined = " ".join(cleaned).strip()
        self._pending_blocks = [joined] if joined else []
        self._pending_line = joined
        return self._pending_blocks

    def _events_for_speech(self, speak: str | list[str], ui: dict[str, Any], state: str, with_audio: bool = True) -> list[TutorEvent]:
        blocks = self._set_speech(speak)
        events = [
            TutorEvent("state", {"state": state, **ui}),
            TutorEvent("ui", ui),
        ]
        for block in blocks:
            events.append(TutorEvent("speak_text", {"text": block}))
        if with_audio:
            audio = self.pending_speech_audio()
            if audio:
                events.append(TutorEvent("audio", {"mime": "audio/wav"}, audio=audio))
        return events

    def start_turn(self) -> list[TutorEvent]:
        self.assembler.reset()
        self.webm_buf.clear()
        self.busy = False
        self.accepting = False
        self.invent_item(allow_llm=False)
        self.state.phase = "prompt"
        self.state.awaiting_completion = True
        parts: list[str] = []
        if not self.introduced:
            parts.append(self.intro_speech())
            self.introduced = True
        parts.append(self.game_brief())
        parts.append(self.prompt_speech())
        ui = self.ui_snapshot({"feedback": f"Your turn, {self.child_name}!"})
        events = self._events_for_speech(" ".join(parts), ui, "speaking", with_audio=False)
        self.state.phase = "listening"
        return events

    def pending_speech_audio(self) -> Optional[bytes]:
        if self.tts is None:
            return None
        blocks = self._pending_blocks or ([self._pending_line] if self._pending_line else [])
        if not blocks:
            return None
        pieces: list[np.ndarray] = []
        sample_rate = 24000
        try:
            for index, block in enumerate(blocks):
                samples, sample_rate = self.tts.synthesize(block, voice=self.voice_id)
                if samples is None or len(samples) == 0:
                    continue
                pieces.append(np.asarray(samples, dtype=np.float32))
                if index < len(blocks) - 1:
                    gap = 0.28
                    pieces.append(np.zeros(int(int(sample_rate) * gap), dtype=np.float32))
        except Exception as exc:
            print(f"TTS failed: {exc}", flush=True)
            return None
        if not pieces:
            return None
        return pcm_to_wav_bytes(np.concatenate(pieces), int(sample_rate))

    def mark_ready(self) -> None:
        self.accepting = True
        self.busy = False
        self.assembler.reset()
        self.state.phase = "listening"

    def _heard_answer(self, transcript: str) -> str:
        cleaned = self._clean_phrase(transcript).lower().strip(" .")
        return cleaned or str(self.current_item().get("expected") or "that")

    def _llm_judge(self, transcript: str) -> Optional[dict[str, Any]]:
        if self.llm is None:
            return None
        item = self.current_item()
        try:
            if self._is_story():
                system = SYSTEM_JUDGE_STORY
                payload = {
                    "sentence": item.get("sentence") or item.get("stem", ""),
                    "target_word": item.get("target") or item.get("expected", ""),
                    "child_said": transcript,
                }
            elif self._is_complete_like():
                system = SYSTEM_JUDGE_COMPLETE
                payload = {
                    "blank": item.get("blank") or item.get("stem", ""),
                    "pos": item.get("pos", "noun"),
                    "tense": item.get("tense", "present"),
                    "example": item.get("expected", ""),
                    "child_said": transcript,
                }
            else:
                system = SYSTEM_JUDGE_MISTAKE
                payload = {
                    "broken": item.get("spoken", ""),
                    "sample_fix": item.get("correct", ""),
                    "kind": item.get("kind", "grammar"),
                    "child_said": transcript,
                }
            raw = self.llm.complete(system, json.dumps(payload), max_tokens=140, temperature=0.1)
            parsed = parse_llm_json(raw)
            if "correct" not in parsed:
                return None
            return {"correct": bool(parsed["correct"]), "next": parsed.get("next"), "speak": parsed.get("speak")}
        except Exception:
            return None

    def evaluate_completion(self, transcript: str) -> dict[str, Any]:
        item = self.current_item()
        expected = item.get("expected", "")
        aliases = [expected] + list(item.get("aliases") or [])
        judged = self._llm_judge(transcript)
        unsafe = is_kid_unsafe(transcript)
        if unsafe:
            ok = False
        elif judged is not None:
            ok = bool(judged["correct"])
        else:
            ok = any(contains_expected(transcript, alias) for alias in aliases if alias)
        word_score = max(similarity(transcript, alias) for alias in aliases if alias) if aliases else 0.0
        phone = max(phoneme_score(transcript, alias) for alias in aliases if alias) if aliases else 0.0
        heard = self._heard_answer(transcript)
        pos = item.get("pos") or "word"
        cheer = random.choice(CHEER_LINES)
        if unsafe:
            speak = f"Let's use kind playground words, {self.child_name}. Try a gentle answer."
            feedback = "Kind words only, please."
        elif ok:
            speak = f"Great job, {self.child_name}! It is indeed {heard}."
            feedback = cheer
        elif phone >= 0.82:
            speak = f"So close, {self.child_name}. Try that {pos} one more time."
            feedback = f"Almost — listen for the {pos}."
        else:
            speak = f"Nice try, {self.child_name}. We need a {pos} that fits."
            feedback = f"Guess the {pos}."
        result = {
            "correct": ok,
            "speak": speak,
            "feedback": feedback,
            "word_score": round(word_score, 3),
            "phoneme_score": round(phone, 3),
        }
        if judged and judged.get("next"):
            result["next"] = judged["next"]
        return result

    def evaluate_story(self, transcript: str) -> dict[str, Any]:
        item = self.current_item()
        target = str(item.get("target") or item.get("expected") or "")
        judged = self._llm_judge(transcript)
        unsafe = is_kid_unsafe(transcript)
        if unsafe:
            ok = False
        elif judged is not None:
            ok = bool(judged["correct"])
        else:
            said = self._clean_phrase(transcript).lower()
            ok = bool(tokenize(transcript)) and said != target and not unsafe
        if unsafe:
            speak = f"Let's keep our words kind, {self.child_name}."
            feedback = "Kind words only, please."
        elif ok:
            speak = f"Yes, {self.child_name}! That means about the same."
            feedback = random.choice(CHEER_LINES)
        else:
            speak = f"Nice try, {self.child_name}. Say another word like {target}."
            feedback = f"Another word for {target}."
        result = {
            "correct": ok,
            "speak": speak,
            "feedback": feedback,
            "word_score": 1.0 if ok else 0.0,
            "phoneme_score": 1.0 if ok else 0.0,
        }
        if judged and judged.get("next"):
            result["next"] = judged["next"]
        return result

    def evaluate_mistake(self, transcript: str) -> dict[str, Any]:
        item = self.current_item()
        said = normalize_text(transcript)
        spoken = item.get("spoken", "")
        correct = item.get("correct", "")
        judged = self._llm_judge(transcript)
        if is_kid_unsafe(transcript):
            ok = False
            issue = "kind-words"
        elif judged is not None:
            ok = bool(judged["correct"])
            issue = "none" if ok else item.get("kind", "grammar")
        else:
            sim_broken = similarity(said, spoken)
            sim_fix = similarity(said, correct)
            need = distinctive_words(spoken, correct)
            said_tokens = set(tokenize(transcript))
            has_fix = all(word in said_tokens for word in need) if need else sim_fix >= 0.92
            stop = {"the", "a", "an", "to", "of", "and", "it", "is", "i", "on", "in", "for", "my", "we", "she", "he", "you"}
            content = [token for token in tokenize(correct) if token not in stop and len(token) > 2]
            covered = sum(1 for token in content if token in said_tokens) / max(len(content), 1)
            repeated_error = sim_broken >= sim_fix and not has_fix
            ok = bool(has_fix and covered >= 0.84 and sim_fix >= 0.7 and not repeated_error)
            issue = "none" if ok else item.get("kind", "grammar")
        if is_kid_unsafe(transcript):
            speak = f"Let's use kind playground words, {self.child_name}."
            feedback = "Kind words only, please."
        elif ok:
            speak = f"Great job, {self.child_name}! That was the right way."
            feedback = f"Yes, {self.child_name} — that’s the fix."
        else:
            speak = f"Not quite, {self.child_name}. Change the tricky word and try again."
            feedback = "Keep going."
        result = {
            "correct": ok,
            "speak": speak,
            "feedback": feedback,
            "issue": issue,
            "word_score": round(similarity(transcript, correct), 3),
            "phoneme_score": round(phoneme_score(transcript, correct), 3),
        }
        if judged and judged.get("next"):
            result["next"] = judged["next"]
        return result

    def _answer_phrase(self) -> str:
        item = self.current_item()
        if self._is_complete_like():
            return str(item.get("expected") or "that word")
        if self._is_story():
            sample = self._clean_phrase(item.get("synonym") or "").lower()
            target = self._clean_phrase(item.get("target") or item.get("expected") or "").lower()
            if sample and sample != target:
                return sample
            return "a different word that means the same"
        return str(item.get("correct") or "the right sentence")

    def _after_failure(self, reason: str, coach: Optional[str] = None) -> list[str]:
        self.state.streak = 0
        self.state.misses += 1
        name = self.child_name
        if not coach:
            if reason == "idk":
                coach = f"That's okay, {name}. Try your best. What do you think it could be?"
            elif reason == "idle":
                coach = f"I didn't hear you, {name}. Give it your best try."
            else:
                coach = f"Nice try, {name}. Have another go."
        if self.state.misses <= NUDGE_LIMIT:
            if self.state.misses == 1:
                return [coach]
            return [f"{coach} I'll say it again.", self.prompt_speech()]
        reveal = f"Nice try, {name}. The answer is {self._answer_phrase()}. Okay, here comes the next one."
        self.invent_item()
        return [reveal, self.prompt_speech()]

    def handle_idle(self) -> list[TutorEvent]:
        if self.busy or not self.accepting:
            return []
        self.busy = True
        self.accepting = False
        try:
            self.state.turns += 1
            blocks = self._after_failure("idle")
            ui = self.ui_snapshot(
                {
                    "feedback": "I'm waiting for your voice.",
                    "correct": False,
                    "heard": "(no answer yet)",
                    "idle": True,
                }
            )
            events = self._events_for_speech(blocks, ui, "speaking", with_audio=False)
            self.state.phase = "listening"
            return events
        finally:
            self.busy = False

    def _apply_next(self, parsed: dict[str, Any], correct: bool) -> None:
        if correct:
            candidate = parsed.get("next") if isinstance(parsed, dict) else None
            item = self._normalize_item(candidate) if candidate else {}
            if self._valid_item(item) and not self._too_similar(item):
                self._commit_item(item)
                return
            self.invent_item()

    def handle_transcript(self, transcript: str, rms: float = 1.0, confidence: float = 1.0) -> list[TutorEvent]:
        transcript = (transcript or "").strip()
        events = [TutorEvent("transcript", {"text": transcript, "partial": False, "heard": transcript})]
        if is_give_up(transcript):
            self.state.turns += 1
            blocks = self._after_failure("idk")
            ui = self.ui_snapshot(
                {
                    "feedback": "Try your best — then we'll learn it together.",
                    "correct": False,
                    "heard": transcript,
                    "give_up": True,
                }
            )
            events.extend(self._events_for_speech(blocks, ui, "speaking", with_audio=False))
            self.state.phase = "listening"
            return events
        if is_garbled(transcript, rms=rms, confidence=confidence):
            speak = self._fresh_line(RETRY_LINES)
            ui = self.ui_snapshot(
                {
                    "feedback": "I didn’t hear that clearly. Try once more.",
                    "heard": transcript or "(too quiet or fuzzy)",
                    "correct": False,
                    "unclear": True,
                }
            )
            events.extend(self._events_for_speech(speak, ui, "speaking", with_audio=False))
            self.state.phase = "listening"
            return events
        self.state.phase = "evaluating"
        if self.state.mode == "mistake":
            rule = self.evaluate_mistake(transcript)
        elif self._is_story():
            rule = self.evaluate_story(transcript)
        else:
            rule = self.evaluate_completion(transcript)
        result = dict(rule)
        self.state.turns += 1
        celebrate = False
        if result["correct"]:
            self.state.streak += 1
            self.state.best = max(self.state.best, self.state.streak)
            celebrate = True
            self._apply_next(result, True)
            coach = str(result.get("speak") or f"Great job, {self.child_name}!")
            if self.state.streak > 0 and self.state.streak % STREAK_REWARD == 0:
                coach = f"Wow, {self.child_name}! That is {self.state.streak} in a row. {coach}"
            blocks = f"{coach} Okay, here comes the next one. {self.prompt_speech()}"
        else:
            blocks = self._after_failure("wrong", str(result.get("speak") or ""))
        ui = self.ui_snapshot(
            {
                "feedback": result.get("feedback") or "",
                "correct": result["correct"],
                "reward": bool(result["correct"]),
                "celebrate": celebrate,
                "word_score": result.get("word_score"),
                "phoneme_score": result.get("phoneme_score"),
                "issue": result.get("issue"),
                "heard": transcript,
            }
        )
        events.extend(self._events_for_speech(blocks, ui, "speaking", with_audio=False))
        self.state.phase = "listening"
        return events

    def transcribe_pcm(self, pcm: np.ndarray) -> tuple[str, float, float]:
        rms = float(np.sqrt(np.mean(np.square(np.asarray(pcm, dtype=np.float32))) + 1e-12)) if pcm is not None and len(pcm) else 0.0
        if self.stt is None:
            return "", 1.0, rms
        text, confidence = self.stt.transcribe(pcm, self.sample_rate)
        return text, confidence, rms

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
            text, confidence, rms = self.transcribe_pcm(pcm)
            return self.handle_transcript(text, rms=rms, confidence=confidence)
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
