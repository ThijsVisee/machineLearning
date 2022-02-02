import os
import sys

from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
from data.data_loader import VoiceData
from os.path import exists


def compile_and_fit(model, data_train, data_val, name, loss, patience=25):
    # if model is already trained and saved, open it
    if exists(f'{name}'):
        return tf.keras.models.load_model(f'{os.getcwd()}/{name}/')
    # unpack data
    x_train, y_train = data_train
    x_val, y_val = data_val
    MAX_EPOCHS = 300

    # compile and fit
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    model.compile(loss=loss, optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])
    model.fit(x=x_train, y=y_train, epochs=MAX_EPOCHS, validation_data=(x_val, y_val), callbacks=[early_stopping])
    # save model
    model.save(name)
    return model


def model(data_train, data_val, input_shape, output_shape, name):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=input_shape, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=output_shape, activation='softmax')
    ])

    return compile_and_fit(model=model, data_train=data_train, data_val=data_val, loss=tf.losses.SparseCategoricalCrossentropy(), name=name)


def dur_model(data_train, data_val, input_shape, output_shape, name):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=input_shape, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=output_shape)
    ])

    return compile_and_fit(model=model, data_train=data_train, data_val=data_val, loss=tf.losses.MeanSquaredError(), name=name)


def main():
    # Check for TensorFlow GPU access
    print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

    # See TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    d = VoiceData()
    df = d.get_nn_data()

    # create train test val split
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    print(f'Shape train data: {train_df.shape}')
    print(f'Shape val data: {val_df.shape}')
    print(f'Shape test data: {test_df.shape}')

    input_shape = train_df['data'][0].shape[0]
    x_train = np.array(train_df['data'].to_list())
    y_train_note = np.array(train_df['note'].to_list())
    y_train_dur = np.array(train_df['duration'].to_list())
    x_val = np.array(val_df['data'].to_list())
    y_val_note = np.array(val_df['note'].to_list())
    y_val_dur = np.array(val_df['duration'].to_list())
    x_test = np.array(test_df['data'].to_list())
    y_test_note = np.array(test_df['note'].to_list())
    y_test_dur = np.array(test_df['duration'].to_list())

    note_model = model(data_train=[x_train, y_train_note], data_val=[x_val, y_val_note],
                       input_shape=input_shape, output_shape=87, name='note_model')
    duration_model = dur_model(data_train=[x_train, y_train_dur], data_val=[x_val, y_val_dur],
                           input_shape=input_shape, output_shape=1, name='duration_model')

    # test performance on test set
    # for idx, sample in enumerate(x_test):
    #     predicted_note = note_model.predict(np.array([sample, ]))
    #     predicted_note = np.argmax(predicted_note)
    #     predicted_dur = duration_model.predict(np.array([sample, ]))
    #     note_p.append(predicted_note)
    #     note_t.append(y_test_note[idx])
    #     dur_p.append(predicted_dur[0][0])
    #     dur_t.append(y_test_dur[idx])
    #     print(predicted_note, y_test_note[idx])
    #     print(f'Error Note: {round(mean_squared_error(note_t, note_p), 1)}     Error Dur: {round(mean_squared_error(dur_t, dur_p), 1)}')

    for i in range(100):
        last_sample = df['data'].iloc[-1]
        predicted_note = note_model.predict(np.array([last_sample, ]))
        predicted_note = np.argmax(predicted_note)
        predicted_dur = duration_model.predict(np.array([last_sample, ]))
        df = d.get_nn_data(p_note=predicted_note, p_dur=int(predicted_dur[0][0]))
    txt = []
    for index, row in df.tail(100).iterrows():
        dur = int(row['duration'])
        for i in range(dur):
            txt.append(str(int(row['note'])))
    textfile = open("../data/file.txt", "w")
    for element in txt:
        textfile.write(element + '\n')
    textfile.close()

    print("done")


if __name__ == '__main__':
    main()
