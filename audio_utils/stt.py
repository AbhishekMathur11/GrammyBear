try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception as _import_err:
    WHISPER_AVAILABLE = False
    print(f"Warning: {_import_err}")
    print("whisper package not found. Install it with: pip install -U openai-whisper")

if WHISPER_AVAILABLE:
    model = whisper.load_model("turbo")
else:
    model = None


def transcribe_audio(audio_data, sample_rate):
    """Transcribe audio data to text using Whisper"""
    if not WHISPER_AVAILABLE or model is None:
        return None
    result = model.transcribe(audio_data, fp16=False, language='en')
    return result["text"].strip()


def is_available():
    """Check if Whisper is available"""
    return WHISPER_AVAILABLE and model is not None