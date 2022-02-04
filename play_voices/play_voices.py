import numpy as np
import math
import time
import os
import sounddevice as sd
import soundfile as sf

'''
create a sound vector for a single voice
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

'''
play a single voice
'''
def play_voice(voice, sampleRate = 10000, stopEarly = 0):

    sound = get_sound_vector(voice, sampleRate)

    sound = np.array(sound)

    sd.play(np.transpose(sound), sampleRate)
    if(stopEarly != 0):
        time.sleep(stopEarly)
        sd.stop()
    else:
        sd.wait()

'''
play all voices
'''
def play_all_voices(data, sampleRate = 10000, stopEarly = 0):

    durationPerSymbol = 1/16
    ticksPerSymbol = math.floor(sampleRate * durationPerSymbol)

    sound = np.zeros(len(data)*ticksPerSymbol)
    for voice in np.transpose(data):
        sound = sound + get_sound_vector(voice, sampleRate)
    
    sound = np.array(sound)

    sd.play(np.transpose(sound), sampleRate)
    if(stopEarly != 0):
        time.sleep(stopEarly)
        sd.stop()
    else:
        sd.wait()



'''
create an audio file from a single voice
'''
def create_audio_file_single_voice(voice, sampleRate = 10000):

    sound = get_sound_vector(voice, sampleRate)

    sound = np.array(sound)

    fName = f'{os.getcwd()}/out/voice{str(voice+1)}.wav'

    sf.write(fName,sound, sampleRate)

'''
create an audio file with all voices
'''
def create_audio_file(data, sampleRate = 10000):

    durationPerSymbol = 1/16
    ticksPerSymbol = math.floor(sampleRate * durationPerSymbol)

    sound = np.zeros(len(data)*ticksPerSymbol)
    for voice in np.transpose(data):
        sound = sound + get_sound_vector(voice, sampleRate)
    
    sound = np.array(sound)

    fName = f'{os.getcwd()}/out/fugue.wav'

    sf.write(fName, np.transpose(sound), sampleRate)

if __name__ == '__main__':

    # F = np.loadtxt(f'{os.getcwd()}/../data/data.txt', usecols=range(4))
    #
    # chosenVoice = 0
    # voice = F[:, chosenVoice]

    #voice = np.loadtxt(f'{os.getcwd()}/../out/voice1.txt')

    voices = np.loadtxt(f'{os.getcwd()}/../data/data.txt', usecols=range(4))

    #voices = []

    #voices.append(np.loadtxt(f'{os.getcwd()}/../out/voice1.txt'))
    #voices.append(np.loadtxt(f'{os.getcwd()}/../out/voice2.txt'))
    #voices.append(np.loadtxt(f'{os.getcwd()}/../out/voice3.txt'))
    #voices.append(np.loadtxt(f'{os.getcwd()}/../out/voice4.txt'))

    #voices = np.transpose(np.array(voices))

    play_all_voices(voices)

    #create_audio_file(voices)