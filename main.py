import numpy as np

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def fit_model():
    return

def main(d, voice, prec):
    VOICE = voice
    INCLUDED_PRECEDING_TIME = prec

    duration_data = [d.encoded_data[VOICE][0].copy()]
    duration_data[0].append(1)
    for idx, data in enumerate(d.encoded_data[VOICE][1:]):
        if data == duration_data[-1][0:5]:
            duration_data[-1][5] += 1
        else:
            duration_data.append(data.copy())
            duration_data[-1].append(1)

    X = []
    y = []
    for idx, data in enumerate(d.encoded_data[VOICE]):
        if idx <= INCLUDED_PRECEDING_TIME:
            continue
        y.append(data)
        X.append(flatten_list(d.encoded_data[VOICE][idx - INCLUDED_PRECEDING_TIME:idx]))
    
    X = np.array(X).T
    y = np.array(y)

    #print(X.shape)
    #print(y.shape)
    model = LinearRegression(X, y, ridge_alpha=0.005)

    # print(flatten_list(d.encoded_data[VOICE][0:INCLUDED_PRECEDING_TIME]))

    pred = model.predict(flatten_list(d.encoded_data[VOICE][len(d.encoded_data[VOICE]) - INCLUDED_PRECEDING_TIME:len(d.encoded_data[VOICE])]))
    print(pred)

    return d.encoded_data[VOICE].append(pred.tolist())
    # pred = np.array([model.predict(X.T[idx]) for idx in range(10)])


if __name__ == '__main__':
    VOICE = 1
    INCLUDED_PRECEDING_TIME = 1024
    dur = 16
    d = VoiceData()
    for i in range(dur):
        model = main(d, VOICE, INCLUDED_PRECEDING_TIME)