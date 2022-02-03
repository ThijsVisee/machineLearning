import numpy as np
import math
import time
import os
import sounddevice as sd
import soundfile as sf

'''
play a single voice
'''

# this code is adapted from the matlab file provided for the project
def get_sound_vector(voice, sampleRate = 10000):

    symbolicLength = len(voice)
    baseFreq = 440
    durationPerSymbol = 1/16
    ticksPerSymbol = math.floor(sampleRate * durationPerSymbol)

    soundvector1 = np.zeros(symbolicLength*ticksPerSymbol)
    currentSymbol = voice[0]

    startSymbolIndex = 1
    x = 1
    for n in voice:
        if not n == currentSymbol:

            stopSymbolIndex = x
            t1 = (startSymbolIndex - 1) * ticksPerSymbol + 1
            t2 = stopSymbolIndex * ticksPerSymbol
            t3 = t2 - t1

            coveredSoundVectorIndices = np.linspace(int(t1), int(t2), num = int(t3))
            toneLength = len(coveredSoundVectorIndices)
            frequency = baseFreq * 2 ** ((currentSymbol-69)/12)
            toneVector = np.zeros(toneLength)

            y = 0
            while y < toneLength:
                toneVector[y] = math.sin(2 * math.pi * frequency * y / sampleRate)
                y = y + 1
            z = 0
            while z < len(coveredSoundVectorIndices):
                c = int(coveredSoundVectorIndices[z])
                soundvector1[c] = toneVector[z]
                z = z + 1
            currentSymbol = n
            startSymbolIndex = x
        x = x + 1

    return soundvector1

def play_voice(voice, sampleRate = 10000):

    sound = get_sound_vector(voice, sampleRate)

    sd.play(sound, 10000)
    time.sleep(50)
    sd.stop()

def create_audio_file(voice, sampleRate = 10000):

    sound = get_sound_vector(voice, sampleRate)

    sf.write('file.wav',sound, 10000)

if __name__ == '__main__':

    # F = np.loadtxt(f'{os.getcwd()}/../data/data.txt', usecols=range(4))
    #
    # chosenVoice = 0
    # voice = F[:, chosenVoice]

    voice = np.loadtxt(f'{os.getcwd()}/../out/voice1.txt')

    play_voice(voice)

    create_audio_file(voice)