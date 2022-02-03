import numpy as np
import os
from analysis.visualization import visualize_single_voice

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression
from analysis.validation import *
from analysis.analysis import get_voice_statistics


def flatten_list(l):
    return [item for sublist in l for item in sublist]

def check_dir_exists(dir):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)

def write_to_file(data, voice):
    fTitle = f'{os.getcwd()}/out/voice{str(voice+1)}.txt'

    check_dir_exists('out')

    with open(fTitle,'w') as f:

        keys = VoiceData.get_voice_from_encoding(data)

        for key in keys:
            f.write(str(key) + '\n')


def get_prediction(d, voice, preceding_notes, pred):
    duration_data = [d.encoded_data[voice][0].copy()]
    duration_data[0].append(1)
    max_duration = 0
    for idx, data in enumerate(d.encoded_data[voice][1:]):
        if data == duration_data[-1][0:5]:
            duration_data[-1][5] += 1
            if duration_data[-1][0] != 0:
                max_duration = max(max_duration, duration_data[-1][5])
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

    # count index of predictions we make
    count = 0

    while idx < pred:
        predicted_pitch, duration = model.predict(flatten_list(duration_data[-preceding_notes:]))

        duration = round(duration) if (((round(duration) % 2) == 0)) else round(duration) + 1


        #duration = round(duration)
        if duration < 1:
            duration = 1
        elif duration > 16:
            duration = 16
        # print("prediction", predicted_pitch, VoiceData.get_pitch_from_absolute(predicted_pitch))
        duration_data.append(VoiceData.encode_from_absolute_pitch(predicted_pitch) + [duration])
        # print("test", VoiceData.encode_from_absolute_pitch(round(predicted_pitch)) + [duration])
        #print(duration_data[-1], VoiceData.get_pitch_from_absolute(duration_data[-1][0]))
        #print(VoiceData.get_pitch_from_absolute(predicted_pitch), duration)
        idx += duration
        count += 1
    
    return duration_data, count



if __name__ == '__main__':

    VOICE = 1

    # values below are multiplied by 16 to get the actual number of notes from bars
    INCLUDED_PRECEDING_STEPS = 12 * 16
    PREDICTION = 24 * 16

    write_all_data = True

    d = VoiceData('data.txt', True)

    get_voice_statistics(d.raw_data[VOICE])

    prediction, predCount = get_prediction(d, VOICE, INCLUDED_PRECEDING_STEPS, PREDICTION)

    prediction = np.array(prediction)

    get_voice_statistics(prediction)

    if(write_all_data):
        write_to_file(prediction, VOICE)
    else:
        write_to_file(prediction[-predCount:], VOICE)

    visualize_single_voice(prediction,VOICE)

    #print(msle(prediction[-predCount:,0],prediction[:predCount,0]))