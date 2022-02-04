import numpy as np
import tensorflow as tf
import os
from analysis.visualization import visualize_single_voice

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression
from model.neural_net import test_performance, predict, write_voice_to_file, nn_model
from analysis.validation import *
from analysis.analysis import get_voice_statistics
from play_voices.play_voices import create_audio_file, play_all_voices, play_voice


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

def neural_network():
    """
    This function trains two neural networks. One to predict the midi note, and another
    to predict the duration of the note. Then 100 new samples are predicted, and written
    to a txt file.
    """
    # Check for TensorFlow GPU access
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

    # See TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # filenames = ['voice1', 'voice2', 'voice3', 'voice4']
    # for idx, filename in enumerate(filenames):
    idx = 1
    vd = VoiceData('data.txt', True)
    df = vd.get_nn_data(idx)

    # create train-test-val split
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    # create model to predict the midi note, and a model to predict the duration
    input_shape = train_df['data'][0].shape[0]
    output_shape_note = len(train_df['note'][0])
    output_shape_duration = len(train_df['duration'][0])

    note_model = nn_model(df_train=train_df, df_val=val_df, input_shape=input_shape, output_shape=output_shape_note,
                          activation='softmax', loss='mean_squared_error', label='note')

    duration_model = nn_model(df_train=train_df, df_val=val_df, input_shape=input_shape, output_shape=output_shape_duration,
                              activation=None, loss='mean_squared_error', label='duration')

    # test the performance of the model on the test set
    test_performance(df_test=test_df, note_model=note_model, duration_model=duration_model)

    # predict new music and append to existing dataset:
    df = predict(df=df, vd=vd, note_model=note_model, duration_model=duration_model)

    # write only the predictions to a file
    write_voice_to_file(df=df, filename=f'voice{idx}')


def ridge_regression(d, voice, preceding_notes, pred):
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
    play_audio = True

    d = VoiceData('data.txt', True)

    allPred = []
    allVoices = []

    for vDx, v in enumerate(d.encoded_data):

        print(f'Predicting Voice {vDx+1}')

        get_voice_statistics(d.raw_data[vDx])

        prediction, predCount = ridge_regression(d, vDx, INCLUDED_PRECEDING_STEPS, PREDICTION)

        neural_network()

        prediction = np.array(prediction)

        get_voice_statistics(prediction)

        if(write_all_data):
            write_to_file(prediction, vDx)
        else:
            write_to_file(prediction[-predCount:], vDx)

        visualize_single_voice(prediction, vDx)

        allPred.append(prediction)
        allVoices.append(VoiceData.get_voice_from_encoding(prediction))

        #play_voice(VoiceData.get_voice_from_encoding(prediction))

    create_audio_file(np.array(allVoices))

    print("creating audio file")

    if(play_audio):
        play_all_voices(np.array(allVoices))

    #print(msle(prediction[-predCount:,0],prediction[:predCount,0]))
