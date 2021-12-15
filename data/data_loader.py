import numpy as np
import os as os


class VoiceData:
    def __init__(self):
        self.data = None
        self.data_path = 'data.txt'
        self.load_voices()

    def load_voices(self):
        cwd = os.getcwd()
        raw_data = np.loadtxt(f'{cwd}/{self.data_path}', dtype=int)
        samples, voices = raw_data.shape
        self.data = np.zeros((voices, samples))
        self.data = [raw_data[:, i] for i in range(voices)]
