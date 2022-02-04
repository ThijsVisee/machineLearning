import numpy as np
import math
import time
from playsound import playsound
import os
import sounddevice as sd

def create_soundvector(F, chosenvoice):

    voice = F[:, chosenvoice]
    print(len(voice))
    #voice = F

    symbolicLength = len(voice)
    baseFreq = 440
    sampleRate = 10000
    durationPerSymbol = 1/16
    ticksPerSymbol = math.floor(sampleRate * durationPerSymbol)

    soundvector1 = np.zeros(symbolicLength*ticksPerSymbol * 10)
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


F = np.loadtxt(f'{os.getcwd()}/../data/generated_voices.txt', usecols=range(4))
#F = np.loadtxt(f'{os.getcwd()}/../data/generated_voice.txt', usecols=range(1))

soundvector1 = create_soundvector(F, 0)
soundvector2 = create_soundvector(F, 1)
soundvector3 = create_soundvector(F, 2)
soundvector4 = create_soundvector(F, 3)

sd.play((soundvector1 + soundvector2 + soundvector3 + soundvector4)/4, 10000)
time.sleep(50)
sd.stop()




