import numpy as np
import tensorflow as tf
import os
from analysis.visualization import plot_error_rate, plot_single_voice

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression
from model.neural_net import test_performance, predict, write_voice_to_file, nn_model
from analysis.validation import msle, get_prob_vec
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
                          activation='softmax', loss='categorical_crossentropy', label='note')

    duration_model = nn_model(df_train=train_df, df_val=val_df, input_shape=input_shape, output_shape=output_shape_duration,
                              activation='softmax', loss='categorical_crossentropy', label='duration')

    # test the performance of the model on the test set
    test_performance(df_test=test_df, note_model=note_model, duration_model=duration_model)

    # predict new music and append to existing dataset:
    df = predict(df=df, vd=vd, note_model=note_model, duration_model=duration_model)

    # write only the predictions to a file
    write_voice_to_file(df=df, filename=f'voice{idx}')


def ridge_regression(d, voice, preceding_notes, pred, ridge_alpha = 0.005):
    raw_data = d.raw_data[voice]
    duration_data = d.get_duration_data(voice)
    min_note, max_note = d.get_min_max_voice_value(voice)

    if preceding_notes >= len(duration_data)-2:
        raise RuntimeError("Excluded steps are higher than length of notes")

    pitches_original = [0] * (max_note - min_note + 2)
    for note in raw_data:
        pitches_original[note - min_note] += 1

    X = []
    y_note = []
    y_duration = []
    idxAdded = 0
    for idx, data in enumerate(duration_data):
        if idx <= preceding_notes:
            continue

        idxAdded += 1
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
    note_model = LinearRegression(X, y_note, ridge_alpha)
    duration_model = LinearRegression(X, y_duration, ridge_alpha)

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

        predicted_duration = np.argmax(predicted_duration)+1
        durations[predicted_duration-1] += 1

        #trim the last value to recieve the same length for all voices
        if idx + predicted_duration > pred:
            predicted_duration = pred-idx

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
        idx += predicted_duration
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


def determine_preceding_steps(d, predLength):

    print('Determine optimal number of preceding steps')

    preceding_steps = np.arange(1,501,1)

    all_errors = []
    all_euclidean = []

    steps_not_too_high = True
    step = 0
    while steps_not_too_high:

        eucDistance = 0
        msle_error = 0

        for vDx, v in enumerate(d.encoded_data):

            try:
                prediction, predCount = ridge_regression(d, vDx, preceding_steps[step], predLength)
            except:
                steps_not_too_high = False
                eucDistance = 1
                msle_error = 1
                break

            prediction = np.array(prediction)

            orig_prob_vec = get_prob_vec(prediction[:-predCount])
            pred_prob_vec = get_prob_vec(prediction[-predCount:])

            # we can reasonably assume that the error will only be 0 after the initialization
            eucDistance = distance.euclidean(orig_prob_vec, pred_prob_vec) if eucDistance == 0 else (eucDistance + distance.euclidean(orig_prob_vec, pred_prob_vec))/2
            msle_error = msle(pred_prob_vec,orig_prob_vec) if msle_error == 0 else (eucDistance + msle(pred_prob_vec,orig_prob_vec))/2

        #print(f"MSLE: {msle_error}")
        #print(f"Euclidean Distance: {eucDistance}")
        #print('')
        all_errors.append(msle_error)
        all_euclidean.append(eucDistance)

        step +=1

        if(step == len(preceding_steps)):
            steps_not_too_high = False
            break

    print(f"Best preceding steps: {np.argmin(all_errors)+1}")
    plot_error_rate("Means Squared Logarithmic Error", "Number of Preceding Steps", "msle", all_errors)
    #print(f"Best Euclidean Distance: {np.argmin(all_euclidean)+1}")

    # we want to flip the array to return larger values preferably if the same error rates are achieved for larger and smaller values
    return np.argmin(all_errors)+1

def determine_ridge_alpha(d,preceding, predLength):

    print('Determine optimal value for ridge alpha')

    alpha_values = np.arange(0.001, 0.901, 0.001)

    all_errors = []
    all_euclidean = []
    for alpha in alpha_values:

        eucDistance = 0
        msle_error = 0

        for vDx, v in enumerate(d.encoded_data):

            prediction, predCount = ridge_regression(d, vDx, preceding, predLength, alpha)

            prediction = np.array(prediction)

            orig_prob_vec = get_prob_vec(prediction[:-predCount])
            pred_prob_vec = get_prob_vec(prediction[-predCount:])

            # we can reasonably assume that the error will only be 0 after the initialization
            eucDistance = distance.euclidean(orig_prob_vec, pred_prob_vec) if eucDistance == 0 else (eucDistance + distance.euclidean(orig_prob_vec, pred_prob_vec))/2
            msle_error = msle(pred_prob_vec,orig_prob_vec) if msle_error == 0 else (eucDistance + msle(pred_prob_vec,orig_prob_vec))/2

        #print(f"MSLE: {msle_error}")
        #print(f"Euclidean Distance: {eucDistance}")
        #print('')
        all_errors.append(msle_error)
        all_euclidean.append(eucDistance)

    print(f"Best ridge alpha: {alpha_values[np.argmin(all_errors)]}")
    plot_error_rate("Means Squared Logarithmic Error", "Ridge Alpha", "alpha", all_errors)
    #print(f"Best Euclidean Distance: {np.argmin(all_euclidean)+1}")


    return alpha_values[np.argmin(all_errors)]


if __name__ == '__main__':

    write_all_data = True
    play_audio = False

    d = VoiceData('data.txt', True)

    # values below are multiplied by 16 to get the actual number of notes from bars
    PREDICTION = 96 * 16

    #INCLUDED_PRECEDING_STEPS = determine_preceding_steps(d, PREDICTION)
    INCLUDED_PRECEDING_STEPS = 88

    #RIDGE_ALPHA = determine_ridge_alpha(d,INCLUDED_PRECEDING_STEPS, PREDICTION)
    RIDGE_ALPHA = 0.203

    allPred = []
    allVoices = []
    eucDistance = 0
    msle_error = 0

    for vDx, v in enumerate(d.encoded_data):

        print(f'Predicting Voice {vDx+1}')

        #get_voice_statistics(d.raw_data[vDx])

        prediction, predCount = ridge_regression(d, vDx, INCLUDED_PRECEDING_STEPS, PREDICTION, RIDGE_ALPHA)

        #neural_network()

        prediction = np.array(prediction)

        #get_voice_statistics(prediction)

        if(write_all_data):
            write_to_file(prediction, vDx)
        else:
            write_to_file(prediction[-predCount:], vDx)

        #plot_single_voice(prediction, vDx, True)

        orig_prob_vec = get_prob_vec(prediction[:-predCount])
        pred_prob_vec = get_prob_vec(prediction[-predCount:])

        # we can reasonably assume that the error will only be 0 after the initialization
        eucDistance = distance.euclidean(orig_prob_vec, pred_prob_vec) if eucDistance == 0 else (eucDistance + distance.euclidean(orig_prob_vec, pred_prob_vec))/2
        msle_error = msle(pred_prob_vec,orig_prob_vec) if msle_error == 0 else (eucDistance + msle(pred_prob_vec,orig_prob_vec))/2

        allPred.append(prediction)
        allVoices.append(VoiceData.get_voice_from_encoding(prediction))

        #play_voice(VoiceData.get_voice_from_encoding(prediction))

    print(f"MSLE: {msle_error}")
    print(f"Euclidean Distance: {eucDistance}")

    print("creating audio file")

    create_audio_file(np.array(allVoices))

    if(play_audio):
        play_all_voices(np.array(allVoices))

    #print(msle(prediction[-predCount:,0],prediction[:predCount,0]))
