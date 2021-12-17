import numpy as np
import os as os


class VoiceData:
    def __init__(self):
        self.raw_data = []
        self.encoded_data = []
        self.data_path = f'{os.getcwd()}/data.txt'
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
                log_abs_pitch = self.__get_log_abs_pitch(note)
                x_chroma, y_chroma = self.__get_x_y_chroma(note)
                x_fifths, y_fifths = self.__get_x_y_fifths(note)
                v = [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths]
                pitch_encoded_voice.append(v)
            self.encoded_data.append(pitch_encoded_voice)
            print(f"Voice {idx} encoded")
        print("Pitch Encoded Successfully!")

    '''
    return logarithm of the absolute pitch
    '''

    def __get_log_abs_pitch(self, note):
        log_abs_pitch = 0
        return log_abs_pitch

    '''
    return x,y coordinates of the position of the note in the chroma circle
    '''

    def __get_x_y_chroma(self, note):
        x = 0
        y = 0
        return x, y

    '''
    return x,y coordinates of the position of the note in the circle of fifths
    '''

    def __get_x_y_fifths(self, note):
        x = 0
        y = 0
        return x, y