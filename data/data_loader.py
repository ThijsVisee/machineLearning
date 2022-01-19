from base64 import encode
import math
import numpy as np
import os as os
import pandas as pd


class VoiceData:
    __radius = 1
    __chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    __c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
    __lowest_note = 35
    __highest_note = 74
    
    def __init__(self):
        self.raw_data = []
        self.encoded_data = []
        self.data_path = f'{os.path.split(__file__)[0]}{os.sep}data.txt'
        self.__lowest_note = 35
        self.__highest_note = 74
        self.__load_voices()
        self.__encode_pitch()

    def __load_voices(self, include_zeroes=False):
        raw_data = np.loadtxt(self.data_path, dtype=int)
        raw_data = np.array([element for element in raw_data if element[0] != 0])
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
                    x_chroma, y_chroma = VoiceData.__get_x_y(note, 'chroma')
                    x_fifths, y_fifths = VoiceData.__get_x_y(note, 'fifths')
                    v = [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths]
                pitch_encoded_voice.append(v)
            self.encoded_data.append(pitch_encoded_voice)
            print(f"Voice {idx} encoded")
        print("Pitch Encoded Successfully!")
        # pd.DataFrame(self.encoded_data).to_csv("file.csv")

    
    @staticmethod
    def encode_single_pitch(pitch):
        if pitch == 0:
            v = [0, 0, 0, 0, 0]
        else:
            log_abs_pitch = 2 * math.log2(math.pow(2, (pitch / 12)) * 440)
            x_chroma, y_chroma = VoiceData.__get_x_y(pitch, 'chroma')
            x_fifths, y_fifths = VoiceData.__get_x_y(pitch, 'fifths')
            v = [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths]
        return v


    @staticmethod
    def encode_from_absolute_pitch(abs_pitch):
        pitch = VoiceData.get_pitch_from_absolute(abs_pitch)
        return VoiceData.encode_single_pitch(pitch)


    '''
    return logarithm of the absolute pitch
    '''

    def __get_log_abs_pitch(self, note):
        # 69 because this is a round integer in Hz
        n = note - 69
        fx = math.pow(2, (n / 12)) * 440

        min_p = 2 * math.log2(math.pow(2, ((VoiceData.__lowest_note - 69) / 12)) * 440)
        max_p = 2 * math.log2(math.pow(2, ((VoiceData.__highest_note - 69) / 12)) * 440)

        log_abs_pitch = 2 * math.log2(fx) - max_p + (max_p - min_p) / 2
        return log_abs_pitch

    # reverse of get_log_abs_pitch
    def get_pitch_from_absolute(log_abs_pitch):

        min_p = 2 * math.log2(math.pow(2, ((VoiceData.__lowest_note - 69) / 12)) * 440)
        max_p = 2 * math.log2(math.pow(2, ((VoiceData.__highest_note - 69) / 12)) * 440)

        return round(math.log2(math.pow(2, ((log_abs_pitch + max_p - (max_p - min_p)/2)/2))/440) * 12 + 69)

    '''
    return x,y coordinates of the position of the note in the chroma circle, or circle of fifths
    '''

    @staticmethod
    def __get_x_y(note, circle):
        note = ((note - 55) % 12)
        if circle == 'chroma':
            angle = (VoiceData.__chroma[note] - 1) * (360 / 12)
        elif circle == 'fifths':
            angle = (VoiceData.__c5[note] - 1) * (360 / 12)
        x = VoiceData.__radius * math.cos(math.degrees(angle))
        y = VoiceData.__radius * math.sin(math.degrees(angle))
        return x, y
