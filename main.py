import numpy as np

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def main(d, voice, preceding_notes):
    duration_data = [d.encoded_data[voice][0].copy()]
    duration_data[0].append(1)
    for idx, data in enumerate(d.encoded_data[voice][1:]):
        if len(data) < 4:
            continue
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
        X.append(flatten_list(d.encoded_data[voice][idx - preceding_notes:idx]))
    
    X = np.array(X).T
    y = np.array(y)

    model = LinearRegression(X, y, ridge_alpha=0.005)

    for idx, data in enumerate(duration_data):
        if idx <= preceding_notes:
            continue
        predicted_pitch = model.predict(flatten_list(d.encoded_data[voice][idx - preceding_notes:idx]))
        print(idx)
        print(d.get_pitch_from_absolute(predicted_pitch[0]))
        print([duration_data[idx][0]] + [duration_data[idx][5]])
        print(predicted_pitch)
        print()

    return d.encoded_data[voice].append(predicted_pitch.tolist())


if __name__ == '__main__':
    VOICE = 1
    INCLUDED_PRECEDING_STEPS = 128
    d = VoiceData()
    # for i in range(dur):
    model = main(d, VOICE, INCLUDED_PRECEDING_STEPS)
