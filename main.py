from faulthandler import dump_traceback
import numpy as np

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def main(d, voice, preceding_notes):
    duration_data = [d.encoded_data[voice][0].copy()]
    duration_data[0].append(1)
    max_duration = 0
    for idx, data in enumerate(d.encoded_data[voice][1:]):
        if data == duration_data[-1][0:5]:
            duration_data[-1][5] += 1
        else:
            duration_data.append(data.copy())
            duration_data[-1].append(1)
    
    X = []
    y = []
    for idx, data in enumerate(duration_data):
        if idx <= preceding_notes:
            continue
        y.append(np.array([data[0], data[5]]))
        X.append(flatten_list(duration_data[idx - preceding_notes:idx]))
    
    X = np.array(X).T
    y = np.array(y)

    model = LinearRegression(X, y, ridge_alpha=0.005)

    idx = 0
    print(duration_data[-1])
    while idx < 230:
        predicted_pitch, duration = model.predict(flatten_list(duration_data[-preceding_notes - 1: -1]))
        duration = round(duration)
        if duration < 1:
            duration = 1
        elif duration > 16:
            duration = 16
        duration_data.append(VoiceData.encode_from_absolute_pitch(round(predicted_pitch)) + [round(duration)])
        print(VoiceData.get_pitch_from_absolute(predicted_pitch), duration)
        idx += duration


if __name__ == '__main__':
    VOICE = 1
    INCLUDED_PRECEDING_STEPS = 16
    d = VoiceData()
    # for i in range(dur):
    model = main(d, VOICE, INCLUDED_PRECEDING_STEPS)

