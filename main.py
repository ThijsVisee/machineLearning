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

    #print(model.predict(flatten_list(d.encoded_data[VOICE][0:INCLUDED_PRECEDING_TIME])))
    predicted_pitch = model.predict(flatten_list(d.encoded_data[VOICE][len(d.encoded_data[VOICE]) - INCLUDED_PRECEDING_TIME:len(d.encoded_data[VOICE])]))
    #print(d.get_pitch_from_absolute(predicted_pitch[0]))
    #print(predicted_pitch)
    # pred = np.array([model.predict(X.T[idx]) for idx in range(10)])

    d.encoded_data[VOICE].append(predicted_pitch.tolist())
    return predicted_pitch.tolist()

if __name__ == '__main__':
    for voice in range (4):
        print('###################')
        # the number of bars you want to include/16
        INCLUDED_PRECEDING_TIME = 2 * 16
        dur = 16
        d = VoiceData()
        for i in range(dur):
            model = main(d, voice, INCLUDED_PRECEDING_TIME)

        fTitle = './out/output'+str(voice+1)+'.txt'
        with open(fTitle,'w') as f:
            for data in d.encoded_data[voice]:
                val = d.get_pitch_from_absolute(data[0]) if data[0] != 0 else data[0]
                f.write(str(val) + '\n')
        
        print('###################')
        print('')