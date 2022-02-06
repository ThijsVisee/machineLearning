import os
import random
import time
import os as os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from os.path import exists
import heapq


def compile_and_fit(model, df_train, df_val, label, loss, patience=12):
    """
    This function compiles and fits a tensorflow model
    :param model: the neural network
    :param df_train: the train dataset
    :param df_val: the validation dataset
    :param label: the label (either 'note' or 'duration'
    :param loss: the loss function
    :param patience: the patience of the training scheme
    :return: a trained model
    """
    load = False
    # if model is already trained and saved, open it
    if exists(f'trained_models{os.sep}{label}_model') and load:
        return tf.keras.models.load_model(f'{os.getcwd()}{os.sep}trained_models{os.sep}{label}_model{os.sep}')

    # unpack data
    x_train, y_train = unpack_data(df=df_train, label=label)
    x_val, y_val = unpack_data(df=df_val, label=label)
    MAX_EPOCHS = 100

    # compile and fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=loss, optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(x=x_train, y=y_train, epochs=MAX_EPOCHS, validation_data=(x_val, y_val), callbacks=[early_stopping])

    # save model
    model.save(f'trained_models{os.sep}{label}_model')
    return model


def nn_model(df_train, df_val, input_shape, output_shape, activation, loss, label):
    """
    this function builds a tensorflow model
    :param df_train: the training dataset
    :param df_val: the validation dataset
    :param input_shape: the input shape of the neural network
    :param output_shape: the output shape of the neural network
    :param activation: the activation function
    :param loss: the loss function
    :param label: the label (either 'note' or 'duration')
    :return: an un-compiled and un-trained model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=input_shape, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=output_shape, activation=activation)
    ])

    return compile_and_fit(model=model, df_train=df_train, df_val=df_val, loss=loss, label=label)


def unpack_data(df, label):
    """
    This function unpacks a pandas dataframe and returns the training data x, y
    :param df: the pandas dataframe holding the dataset
    :param label: the label (either 'note' or 'duration')
    :return: the train{os.sep}val{os.sep}test data x, y
    """
    x = np.array(df['data'].to_list())
    y = np.array(df[label].to_list())
    return x, y


def test_performance(df_test, note_model, duration_model):
    """
    This function tests the trained model on the test data
    :param df_test: the pandas dataframe holding the test data
    :param note_model: the model that is trained to predict the note
    :param duration_model: the model that is trained to predict the duration
    """
    x_test, y_test_note = unpack_data(df=df_test, label='note')
    _, y_test_dur = unpack_data(df=df_test, label='duration')
    note_p, note_t, dur_p, dur_t = ([] for i in range(4))
    print("TESTING ON TEST DATA:")
    # test performance on test set
    for idx, sample in enumerate(x_test):
        predicted_note = note_model.predict(np.array([sample, ]))
        predicted_note = np.argmax(predicted_note[0])
        predicted_dur = duration_model.predict(np.array([sample, ]))
        predicted_dur = np.argmax(predicted_dur[0])

        note_p.append(predicted_note)
        note_t.append(np.argmax(y_test_note[idx]))
        dur_p.append(predicted_dur)
        dur_t.append(np.argmax(y_test_dur[idx]))
        print(f'Predicted note: {predicted_note} Real Note: {np.argmax(y_test_note[idx])} '
              f'Error: {round(mean_squared_error(note_t, note_p), 1)} '
              f'Predicted duration: {predicted_dur} Real duration: {np.argmax(y_test_dur[idx])} '
              f'Error: {round(mean_squared_error(dur_t, dur_p), 1)}')


def predict(df, vd, note_model, duration_model, num_predictions=100, a=0.0, plot=False):
    """
    This function predicts a given amount of new samples and appends it
    to a copy of the original dataset
    :param df: the original dataset
    :param vd: instance of the VoiceData class
    :param note_model: the model that is trained to predict the note
    :param duration_model: the model that is trained to predict the duration
    :param num_predictions: the number of prediction that should be made
    :param a: alpha, the chance of selecting the second best note
    :param plot: plot the prediction distribution
    :return: the dataset containing the original samples + the predicted samples
    """
    print(f'PREDICTING {num_predictions} NOTES and DURATIONS:')
    remove_last_n_samples = 250
    predictions = []
    for i in range(num_predictions):
        last_sample = df['data'].iloc[-remove_last_n_samples]
        predicted_note = note_model.predict(np.array([last_sample, ]))[0]
        if plot:
            plot_prediction_dist(predicted_note)
        # select the highest predicted note with 90% chance, else second highest
        predicted_note = heapq.nlargest(2, range(len(predicted_note)), key=predicted_note.__getitem__)
        predicted_note = predicted_note[0] if random.random() > a else predicted_note[1]
        predicted_dur = duration_model.predict(np.array([last_sample, ]))
        predicted_dur = np.argmax(predicted_dur[0])
        for dur in range(predicted_dur):
            predictions.append(predicted_note)
        print('Predicted note: ', predicted_note, 'Predicted duration: ', predicted_dur)
        df = vd.get_nn_data(p_note=predicted_note, p_dur=predicted_dur, remove_last_n_samples=remove_last_n_samples)
    write_to_file(predictions)
    return df


def plot_prediction_dist(prediction):
    x = prediction
    y = [i for i in range(87)]
    plt.bar(y, x, align='center')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.show()
    time.sleep(5)
    plt.close()


def write_voice_to_file(df, filename='generated_voice'):
    """
    This function writes the (last 100) predictions to a txt file
    :param df: the dataset containing the original samples + the predicted samples
    :param filename: the name the generated file should have
    """
    txt = []
    for index, row in df.tail(100).iterrows():
        dur = int(np.argmax(row['duration']))
        for i in range(dur):
            txt.append(str(int(np.argmax(row['note']))))
    textfile = open(f'data{os.sep}{filename}.txt', "w")
    for element in txt:
        textfile.write(element + '\n')
    textfile.close()
    print(f'Voice has been written to {filename}.txt')


def write_to_file(predictions, filename='generated_voice'):
    """
    This function writes the (last 100) predictions to a txt file
    :param df: the dataset containing the original samples + the predicted samples
    :param filename: the name the generated file should have
    """
    txt = []
    for prediction in predictions:
        txt.append(str(prediction))
    textfile = open(f'data{os.sep}{filename}.txt', "w")
    for element in txt:
        textfile.write(element + '\n')
    textfile.close()
    print(f'Voice has been written to {filename}.txt')


def write_4_voices_to_file(filename = 'generated_voices'):
    F = np.loadtxt(f'{os.getcwd()}/data/generated_voice.txt', usecols=range(1))


    voice_1 = np.loadtxt(f'{os.getcwd()}/predictions/voice1.txt', usecols=range(1))
    voice_2 = np.loadtxt(f'{os.getcwd()}/predictions/voice2.txt', usecols=range(1))
    voice_3 = np.loadtxt(f'{os.getcwd()}/predictions/voice3.txt', usecols=range(1))
    voice_4 = np.loadtxt(f'{os.getcwd()}/predictions/voice4.txt', usecols=range(1))

    textfile = open(f'data{os.sep}{filename}.txt', "w")
    i = 0
    for n in voice_4:
        textfile.write(str(int(voice_1[i])) + ' ' + str(int(voice_2[i])) + ' ' + str(int(voice_3[i])) + ' ' + str(int(voice_4[i]))+ '\n')
        i = i + 1

    textfile.close()





