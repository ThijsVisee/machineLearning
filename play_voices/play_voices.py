import numpy as np
import math
import time
from playsound import playsound
import os
import sounddevice as sd

#F = np.loadtxt(f'{os.getcwd()}/../data/data.txt', usecols=range(4))
F = np.loadtxt(f'{os.getcwd()}/../out/output.txt')

chosenVoice = 0
voice = F

symbolicLength = len(voice)
baseFreq = 440
sampleRate = 10000
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

sd.play(soundvector1, 10000)
time.sleep(50)
sd.stop()




