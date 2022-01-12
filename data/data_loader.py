import math
import numpy as np
import os as os
import pandas as pd


class VoiceData:
    def __init__(self):
        self.raw_data = []
        self.encoded_data = []
        self.data_path = f'{os.path.split(__file__)[0]}{os.sep}data.txt'
        self.__radius = 1
        self.__chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.__c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
        self.__lowest_note = 35
        self.__highest_note = 74
        self.__load_voices()
        self.__encode_pitch()

    def __load_voices(self):
        raw_data = np.loadtxt(self.data_path, dtype=int)
        samples, voices = raw_data.shape
        self.raw_data = np.array([raw_data[:, i] for i in range(voices)])
        print("Raw Data Loaded Successfully!")

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

    def __encode_pitch(self):
        for idx, voice in enumerate(self.raw_data):
            pitch_encoded_voice = []
            for note in voice:
                if note == 0:
                    v = [0, 0, 0, 0, 0]
                else:
                    log_abs_pitch = self.__get_log_abs_pitch(note)
                    x_chroma, y_chroma = self.__get_x_y(note, 'chroma')
                    x_fifths, y_fifths = self.__get_x_y(note, 'fifths')
                    v = [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths]
                pitch_encoded_voice.append(v)
            self.encoded_data.append(pitch_encoded_voice)
            print(f"Voice {idx} encoded")
        print("Pitch Encoded Successfully!")
        # pd.DataFrame(self.encoded_data).to_csv("file.csv")

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

    '''
    return x,y coordinates of the position of the note in the chroma circle, or circle of fifths
    '''

    def __get_x_y(self, note, circle):
        note = ((note - 55) % 12)
        if circle == 'chroma':
            angle = (self.__chroma[note] - 1) * (360 / 12)
        elif circle == 'fifths':
            angle = (self.__chroma[note] - 1) * (360 / 12)
        x = self.__radius * math.sin(math.degrees(angle))
        y = self.__radius * math.sin(math.degrees(angle))
        return x, y
