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
        self.duration_data = []
        self.data_path = f'{os.path.split(__file__)[0]}{os.sep}data.txt'
        self.__lowest_note = 35
        self.__highest_note = 74
        self.__load_voices()
        self.__encode_pitch()

    def __load_voices(self, include_zeroes=False):
        """
        Loads the voices from a txt file
        :param include_zeroes: bool to determine if zeros should be included in the data
        :return: the raw data
        """
        raw_data = np.loadtxt(self.data_path, dtype=int)
        raw_data = np.array([element for element in raw_data if element[0] != 0])
        samples, voices = raw_data.shape
        self.raw_data = np.array([raw_data[:, i] for i in range(voices)])
        print("Raw Data Loaded Successfully!")

    def __encode_pitch(self):
        """
        Encode pitch of each note in 5dim vector:
        [
        0. value proportional to the logarithm of the absolute pitch of the note
        1. x coordinates of the position of the note in the chroma circle
        2. y coordinates of the position of the note in the chroma circle
        3. x coordinates of the note in the circle of fifths
        4. y coordinates of the note in the circle of fifths
        ]
        :return: encoded pitch
        """
        for idx, voice in enumerate(self.raw_data):
            pitch_encoded_voice = []
            for note in voice:
                if note == 0:
                    v = [0, 0, 0, 0, 0]
                else:
                    v = self.__encode_single_sample(note)
                pitch_encoded_voice.append(v)
            self.encoded_data.append(pitch_encoded_voice)
            print(f"Voice {idx} encoded")
        print("Pitch Encoded Successfully!")

    def __encode_single_sample(self, sample):
        """
        Encodes a single note as a 5 dimensional vector.
        :param sample: the note
        :return: a 5 dim vector of the form [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths]
        """
        log_abs_pitch = self.__get_log_abs_pitch(sample)
        x_chroma, y_chroma = VoiceData.__get_x_y(sample, 'chroma')
        x_fifths, y_fifths = VoiceData.__get_x_y(sample, 'fifths')
        return [log_abs_pitch, x_chroma, y_chroma, x_fifths, y_fifths]

    def get_nn_data(self, p_note=None, p_dur=None, preceding_notes=60):
        """
        This function generates the pandas dataframe containing the train/test/val data.
        If p_note is not passed then this function assumes that the data
        is generated for the first time. If p_note is passed, it is converted to the
        appropriate format and appended to the data. Same for p_dur.
        :param p_note: predicted note
        :param p_dur: predicted duration
        :param preceding_notes: the number of preceding notes is one data window
        :return:
        """
        voice = 1
        if not self.duration_data:
            self.duration_data = [self.encoded_data[voice][0].copy()]
            self.duration_data[0].append(1)
            note_vector = [0.0] * 87
            note_vector[self.raw_data[voice][0]] = 1.0
            self.duration_data[0].append(note_vector)
            max_duration = 0
            for idx, data in enumerate(self.encoded_data[voice][1:]):
                # we also append the midi note to the data, these will be the labels for the NN
                note = int(self.raw_data[voice][idx])
                if data == self.duration_data[-1][0:5]:
                    self.duration_data[-1][5] += 1
                    if self.duration_data[-1][0] != 0:
                        max_duration = max(max_duration, self.duration_data[-1][5])
                else:
                    self.duration_data.append(data.copy())
                    self.duration_data[-1].append(1)
                    note_vector = [0.0] * 87
                    note_vector[note] = 1.0
                    self.duration_data[-1].append(note_vector)

            for item in self.duration_data:
                duration_vector = [0] * 25
                index = item[5] if item[5] <= 24 else 0
                duration_vector[index] = 1
                item[5] = duration_vector

        if p_note:
            # encode the note and duration as a probability vector
            note_vector = [0.0] * 87
            duration_vector = [0] * 25
            note_vector[p_note] = 1.0
            duration_vector[p_dur] = 1.0

            v = self.__encode_single_sample(p_note) + [duration_vector] + [note_vector]
            self.duration_data.append(v)

        # copy the duration data so as to not disturb its values
        data = self.duration_data.copy()
        data = pd.DataFrame(data)
        data = data.rename(columns={0: 'log pitch', 1: 'chroma x', 2: 'chroma y', 3: 'fifths x',
                                    4: 'fifths y', 5: 'duration', 6: 'note'})
        # normalize data
        subset = ['log pitch', 'chroma x', 'chroma y', 'fifths x', 'fifths y']
        data[subset] = (data[subset] - data[subset].mean()) / data[subset].std()

        # Window the data
        window_data = {'data': [], 'duration': [], 'note': []}
        for idx, row in data.iterrows():
            if idx <= preceding_notes:
                continue

            input_data = data[idx - preceding_notes:idx][
                ['log pitch', 'chroma x', 'chroma y', 'fifths x', 'fifths y']]
            input_data = np.array(input_data).flatten()
            window_data['data'].append(input_data)
            window_data['duration'].append(row['duration'])
            window_data['note'].append(row['note'])

        return pd.DataFrame(window_data)

    @staticmethod
    def encode_single_pitch(pitch):
        if pitch == 0:
            v = [0, 0, 0, 0, 0]
        else:
            log_abs_pitch = VoiceData.__get_abs_pitch(pitch)
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

    @staticmethod
    def __get_abs_pitch(note):
        # 69 because this is a round integer in Hz
        n = note - 69
        fx = math.pow(2, (n / 12)) * 440

        min_p = 2 * math.log2(math.pow(2, ((VoiceData.__lowest_note - 69) / 12)) * 440)
        max_p = 2 * math.log2(math.pow(2, ((VoiceData.__highest_note - 69) / 12)) * 440)

        log_abs_pitch = 2 * math.log2(fx) - max_p + (max_p - min_p) / 2
        return log_abs_pitch

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

        return round(math.log2(math.pow(2, ((log_abs_pitch + max_p - (max_p - min_p) / 2) / 2)) / 440) * 12 + 69)

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

    @staticmethod
    def __flatten_list(l):
        return [item for sublist in l for item in sublist]
