import sounddevice as sd
import numpy as np


def record_audio(duration=4, fs=16000):
    """
    Record audio from microphone

    Args:
        duration: Recording duration in seconds
        fs: Sample rate in Hz

    Returns:
        tuple: (audio_array, sample_rate)
    """
    print("Listening.....")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    return np.squeeze(recording), fs


def play_audio(audio_data, sample_rate):
    """
    Play audio data through speakers

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
    """
    print("Playing back recorded input...")
    sd.play(audio_data, sample_rate)
    sd.wait()


def save_audio_to_file(audio_data, sample_rate, filename):
    """
    Save audio data to WAV file

    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        filename: Output filename
    """
    import soundfile as sf
    sf.write(filename, audio_data, sample_rate)
    print(f"Audio saved to {filename}")