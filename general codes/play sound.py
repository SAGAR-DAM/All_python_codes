import numpy as np
import sounddevice as sd
import random
import time

def play_random_sound():
    frequency = random.uniform(200, 1000)  # Random frequency between 200 and 1000 Hz
    duration = random.uniform(0.5, 2.0)  # Random duration between 0.5 and 2.0 seconds

    # Generate a sine wave
    t = np.linspace(0, duration, int(duration * 44100), endpoint=False)  # 44100 samples per second
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    print(wave)
    # Play the sound
    sd.play(wave, samplerate=44100)
    sd.wait()  # Wait for the sound to finish

# Example: Play a random sound
play_random_sound()

