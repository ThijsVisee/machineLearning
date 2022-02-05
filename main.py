import numpy as np
import tensorflow as tf
import os
from analysis.visualization import boxplot, plot_all_voices, plot_single_voice

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression
from model.neural_net import test_performance, predict, write_voice_to_file, nn_model
from analysis.validation import *
from analysis.analysis import get_voice_statistics
from analysis.visualization import plot_single_voice
from play_voices.play_voices import create_audio_file, play_all_voices

from scipy.spatial import distance

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
    raw_data = d.raw_data[voice]
    duration_data = d.get_duration_data(voice)
    min_note, max_note = d.get_min_max_voice_value(voice)

    pitches_original = [0] * (max_note - min_note + 2)
    for note in raw_data:
        pitches_original[note - min_note] += 1
    
    pitches_dur = [0] * (max_note - min_note + 2)
    #for dur_note in duration_data:
        #pitches

    X = []
    y_note = []
    y_duration = []
    for idx, data in enumerate(duration_data):
        if idx <= preceding_notes:
            continue

        y_note.append([0] * (max_note - min_note + 2))
        y_duration.append([0] * 16)
        y_duration[-1][min(data[5], 16) - 1] += 1
        y_note[-1][round(VoiceData.get_pitch_from_absolute(data[0])) - min_note] += 1
        X.append(flatten_list(duration_data[idx - preceding_notes:idx]))

    # Create input and output data
    X = np.array(X).T
    y_duration = np.array(y_duration)
    y_note = np.array(y_note)

    # Train the linear regression model
    note_model = LinearRegression(X, y_note, ridge_alpha=0.005)
    duration_model = LinearRegression(X, y_duration, ridge_alpha=0.005)

    previous_pitch = -1
    durations = [0] * 16
    pitches = [0] * (max_note - min_note + 2)

    idx = 0

    # count index of predictions we make
    count = 0

    while idx < pred:
        # Perform the prediction
        input = flatten_list(duration_data[-preceding_notes:])
        predicted_pitch = note_model.predict(input)
        predicted_duration = duration_model.predict(input)

        predicted_duration = np.argmax(predicted_duration) 
        durations[predicted_duration] += 1


        if predicted_pitch[-1] == np.max(predicted_pitch):
            predicted_pitch = 0
        else:
            max_args = np.argsort(predicted_pitch)
            predicted_pitch = max_args[0] + min_note
            if predicted_pitch == previous_pitch:
                predicted_pitch = max_args[1] + min_note
                
            pitches[predicted_pitch - min_note] += predicted_duration
            previous_pitch = predicted_pitch

        # Append the encoded predicted pitch and duration to the data to use it as input for the next prediction
        duration_data.append(VoiceData.encode_single_pitch(predicted_pitch) + [predicted_duration])

        # print("test", VoiceData.encode_from_absolute_pitch(round(predicted_pitch)) + [duration])
        #print(duration_data[-1], VoiceData.get_pitch_from_absolute(duration_data[-1][0]))
        #print(VoiceData.get_pitch_from_absolute(predicted_pitch), duration)
        idx += predicted_duration + 1
        count += 1
    
    return duration_data, count

    #pitches_original = [x/sum(pitches_original) for x in pitches_original]
    #pitches = [x/sum(pitches) for x in pitches]
    #for x in pitches_original:
    #    print(f"{x: .2f} ", end="")
    #print()
    #for x in pitches:
    #    print(f"{x: .2f} ", end="")
    #print()
    #return distance.euclidean(pitches_original, pitches)



if __name__ == '__main__':

    VOICE = 1

    # values below are multiplied by 16 to get the actual number of notes from bars
    INCLUDED_PRECEDING_STEPS = 12 * 16
    PREDICTION = 24 * 16

    write_all_data = True
    play_audio = False

    d = VoiceData('data.txt', True)

    allPred = []
    allVoices = []

    for vDx, v in enumerate(d.encoded_data):

        print('###########')

        get_voice_statistics(d.raw_data[vDx],False)

        prediction, predCount = ridge_regression(d, vDx, INCLUDED_PRECEDING_STEPS, PREDICTION)

        print(f'Predicting Voice {vDx+1}')

        #neural_network()

        prediction = np.array(prediction)

        get_voice_statistics(prediction, False)

        if(write_all_data):
            write_to_file(prediction, vDx)
        else:
            write_to_file(prediction[-predCount:], vDx)

        plot_single_voice(prediction[:-predCount], vDx, False)

        allPred.append(prediction[:-predCount])
        allVoices.append(VoiceData.get_voice_from_encoding(prediction))

        #play_voice(VoiceData.get_voice_from_encoding(prediction))
    
    plot_all_voices(allPred, False)

    boxplot(allPred, True)

    #create_audio_file(np.array(allVoices))

    print("creating audio file")

    if(play_audio):
        play_all_voices(np.array(allVoices))

    #print(msle(prediction[-predCount:,0],prediction[:predCount,0]))
