from faulthandler import dump_traceback
import numpy as np

from data.data_loader import VoiceData
from model.linear_regression import LinearRegression


from scipy.spatial import distance
from tqdm import tqdm

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def main(d, voice, preceding_notes):
    raw_data = d.raw_data[voice]
    duration_data = d.get_duration_data(voice)
    min_note, max_note = d.get_min_max_voice_value(voice)

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
    X = np.array(X)
    y_duration = np.array(y_duration)
    y_note = np.array(y_note)

    # Train the linear regression model
    note_model = LinearRegression(X, y_note, ridge_alpha=0.005)
    duration_model = LinearRegression(X, y_duration, ridge_alpha=0.001)

    with open("out/prediction.txt", "w") as f:
        # for x in raw_data[3000:]:
        #     f.write(f"{x}\n")

        idx = 0
        previous_pitch = -1
        while idx < 1500:
            # Perform the prediction
            input = flatten_list(duration_data[-preceding_notes - 1: -1])
            predicted_pitch = note_model.predict(input)
            predicted_duration = duration_model.predict(input) 

            predicted_duration = np.argmax(predicted_duration) 
            predicted_duration += 1
            predicted_duration *= 2

            if predicted_pitch[-1] == np.max(predicted_pitch):
                predicted_pitch = 0
            else:
                max_args = np.argsort(predicted_pitch)
                predicted_pitch = max_args[0] + min_note
                if predicted_pitch == previous_pitch:
                    predicted_pitch = max_args[1] + min_note
                
                previous_pitch = predicted_pitch

            print(f"Predicted pitch: {predicted_pitch}\tduration: {predicted_duration}")

            # Append the encoded predicted pitch and duration to the data to use it as input for the next prediction
            duration_data.append(VoiceData.encode_from_absolute_pitch(predicted_pitch) + [predicted_duration*2])

            for _ in range(predicted_duration*2):
                f.write(str(predicted_pitch))
                f.write("\n")

            idx += predicted_duration + 1
    



if __name__ == '__main__':
    VOICE = 0
    INCLUDED_PRECEDING_STEPS = [32]
    d = VoiceData(VOICE)

    max_cos = 100
    max_idx = -1
    for x in tqdm(INCLUDED_PRECEDING_STEPS): 
        result = main(d, VOICE, x)
        print(f"{x}: {result}")
        # if result < max_cos:
        #     max_cos = result
        #     max_idx = x
    
    print(f"Max cos: {max_cos} at {max_idx}")