from pathlib import Path
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro
import tempfile
import os


MODEL_DIR = Path(__file__).parent / "tts" / "kokoro-tts" / "models"
DEFAULT_VOICE = "af_sarah"  # Changed to a more standard voice
DEFAULT_LANG = "en-us"
DEFAULT_SPEED = 1.0

# Initialize Kokoro model once
_kokoro_model = None
_model_path = MODEL_DIR / "kokoro-v1.0.onnx"
_voices_path = MODEL_DIR / "voices-v1.0.bin"


def _get_kokoro():
    """Get or initialize the Kokoro model"""
    global _kokoro_model
    if _kokoro_model is None:
        if not _model_path.exists():
            raise FileNotFoundError(f"Model file not found: {_model_path}")
        if not _voices_path.exists():
            raise FileNotFoundError(f"Voices file not found: {_voices_path}")
        _kokoro_model = Kokoro(str(_model_path), str(_voices_path))
    return _kokoro_model


def text_to_speech(text: str, output: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED, lang: str = DEFAULT_LANG):
    """Convert text to speech and save to file"""
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    # Save as WAV file
    import soundfile as sf
    sf.write(output, samples, sample_rate)


def stream_text(text: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED, lang: str = DEFAULT_LANG):
    """Stream text to speech directly to audio output"""
    kokoro = _get_kokoro()
    samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    # Play audio directly
    sd.play(samples, sample_rate)
    sd.wait()


def speak(text: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED, lang: str = DEFAULT_LANG):
    """Convenience function to speak text"""
    stream_text(text, voice=voice, speed=speed, lang=lang)


def list_voices():
    """List available voices"""
    kokoro = _get_kokoro()
    voices = kokoro.get_voices()
    print("Available voices:")
    for idx, voice in enumerate(voices):
        print(f"{idx + 1}. {voice}")
    return voices


def list_languages():
    """List available languages"""
    kokoro = _get_kokoro()
    languages = kokoro.get_languages()
    print("Available languages:")
    for idx, lang in enumerate(languages):
        print(f"{idx + 1}. {lang}")
    return languages


def save_audio(text: str, output_file: str, voice: str = DEFAULT_VOICE):
    """Save audio to file (alias for text_to_speech for compatibility)"""
    text_to_speech(text, output_file, voice=voice)