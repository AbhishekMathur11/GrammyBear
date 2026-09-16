# test_audio.py
import sys
from pathlib import Path

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from audio_utils.core import record_audio, play_audio
from audio_utils.tts import speak
from audio_utils.stt import transcribe_audio, is_available


def main():
    print("=== Continuous STT/TTS Test ===")
    if not is_available():
        print("Whisper not available - install with: pip install openai-whisper")
    print("Sequence: Listen → Record/Playback → Transcribe → Speak Back")
    print("Listening duration: 8 seconds per cycle")
    print("Press Ctrl+C to stop\n")

    # Initial verification
    speak("System ready")

    try:
        while True:
            # 1. Listen and Record (8 seconds for better phrase capture)
            print("🎤 Listening for 8 seconds...")
            audio_array, fs = record_audio(duration=8, fs=16000)

            # Check if we got meaningful audio
            if np.max(np.abs(audio_array)) < 0.01:
                print("⚠️  Very quiet - skipping this cycle")
                continue

            # 2. Play back the recorded audio
            print("🔊 Playing back your audio...")
            play_audio(audio_array, fs)

            # 3. Transcribe the audio
            if is_available():
                print("📝 Transcribing...")
                text = transcribe_audio(audio_array, fs)
                if text and text.strip():
                    print(f"✅ You said: \"{text}\"")
                    # 4. Speak back what was said
                    speak(f"You said: {text}")
                else:
                    print("❓ No speech detected in audio")
                    speak("I didn't catch that")
            else:
                print("⚠️  Whisper not available for transcription")

            print("-" * 50)  # Visual separator between cycles

    except KeyboardInterrupt:
        print("\n🛑 Stopping test...")
        speak("Test stopped")


if __name__ == "__main__":
    main()