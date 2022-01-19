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

    idx == len(duration_data) - 1
    print(duration_data[-1])
    while idx < 1300:
        predicted_pitch, duration = model.predict(flatten_list(d.encoded_data[voice][-preceding_notes-1:-1]))
        predicted_pitch = d.get_pitch_from_absolute(predicted_pitch)
        print(f"{idx}: {predicted_pitch}, {duration}")
        if duration > 12:
            break
        for _ in range(round(duration)) if duration >= 0.5 else range(1):
            d.encoded_data[voice].append(VoiceData.encode_single_pitch(predicted_pitch))
            idx += 1
            print(d.encoded_data[voice][-1])
            print(d.encoded_data[voice][-2])


    return d.encoded_data[voice].append(predicted_pitch.tolist())


if __name__ == '__main__':
    VOICE = 1
    INCLUDED_PRECEDING_STEPS = 16
    d = VoiceData()
    # for i in range(dur):
    model = main(d, VOICE, INCLUDED_PRECEDING_STEPS)
