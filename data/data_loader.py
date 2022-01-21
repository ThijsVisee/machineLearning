import math
import numpy as np
import os as os
import pandas as pd


class VoiceData:
    def __init__(self):
        self.raw_data = []
        self.encoded_data = []
        self.numVoices = 0
        self.data_path = f'{os.path.split(__file__)[0]}{os.sep}data.txt'
        self.__radius = 1
        self.__chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.__c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
        self.__lowest_note = 0
        self.__highest_note = 88
        self.__load_voices()
        self.__encode_voices()

    def __load_voices(self):
        raw_data = np.loadtxt(self.data_path, dtype=int)
        samples, voices = raw_data.shape
        self.numVoices = voices
        self.raw_data = np.array([raw_data[:, i] for i in range(voices)])
        print("Raw Data Loaded Successfully!")

    def __encode_voices(self):
        for idx, voice in enumerate(self.raw_data):
            self.encoded_data.append(self.encode_pitch(voice, idx))
            print(f"Voice {idx} encoded")
        print("Pitch Encoded Successfully!")


    '''
    
        Encode pitch of each note in 5dim vector:
    [
    0. value proportional to the logarithm of the absolute pitch of the note
    1. x coordinates of the position of the note in the chroma circle
    2. y coordinates of the position of the note in the chroma circle
    3. x coordinates of the note in the circle of fifths
    4. y coordinates of the note in the circle of fifths
    ]
    '''

    def encode_pitch(self, voice, idx):
        pitch_encoded_voice = []

        for ndx, note in enumerate(voice):

            dist1, dist2, dist3 = self.__get_distances(self.raw_data[:,ndx], idx)

            if note == 0:
                v = [0, 0, 0, 0, 0, dist1, dist2, dist3]
            else:
                log_abs_pitch = self.__get_log_abs_pitch(note)
                x_chroma, y_chroma = self.__get_x_y(note, 'chroma')
                x_fifths, y_fifths = self.__get_x_y(note, 'fifths')
                v = [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths, dist1, dist2, dist3]

            pitch_encoded_voice.append(v)

        return pitch_encoded_voice

    '''
    return logarithm of the absolute pitch
    '''

    def __get_log_abs_pitch(self, note):
        # 69 because this is a round integer in Hz
        n = note - 69
        fx = math.pow(2, (n / 12)) * 440

        min_p = 2 * math.log2(math.pow(2, ((self.__lowest_note - 69) / 12)) * 440)
        max_p = 2 * math.log2(math.pow(2, ((self.__highest_note - 69) / 12)) * 440)

        log_abs_pitch = 2 * math.log2(fx) - max_p + (max_p - min_p) / 2
        return log_abs_pitch

    # reverse of get_log_abs_pitch
    def get_pitch_from_absolute(self, log_abs_pitch):

        min_p = 2 * math.log2(math.pow(2, ((self.__lowest_note - 69) / 12)) * 440)
        max_p = 2 * math.log2(math.pow(2, ((self.__highest_note - 69) / 12)) * 440)

        return round(math.log2(math.pow(2, ((log_abs_pitch + max_p - (max_p - min_p)/2)/2))/440) * 12 + 69)

    '''
    return x,y coordinates of the position of the note in the chroma circle, or circle of fifths
    '''

    def __get_x_y(self, note, circle):
        note = ((note - 55) % 12)
        if circle == 'chroma':
            angle = (self.__chroma[note] - 1) * (360 / 12)
        elif circle == 'fifths':
            angle = (self.__c5[note] - 1) * (360 / 12)
        x = self.__radius * math.cos(math.degrees(angle))
        y = self.__radius * math.sin(math.degrees(angle))
        return x, y


    '''
    return the distance of the note to the notes in the other three voices
    '''
    def __get_distances(self, notes, vIdx):
        dist = []
        for idx, note in enumerate(notes):
            if idx != vIdx:
                dist.append(note - notes[vIdx])

        return dist