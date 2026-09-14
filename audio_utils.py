import sounddevice as sd
import pyttsx3
import numpy as np


def speak(text):

    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def record_audio(duration=4, fs=16000):

    print("Listening .....")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    return np.squeeze(recording), fs