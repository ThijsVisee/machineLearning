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
    # print(min_note, max_note)

    pitches_original = [0] * (max_note - min_note + 2)
    for note in raw_data:
        pitches_original[note - min_note] += 1
    
    pitches_dur = [0] * (max_note - min_note + 2)
    for dur_note in duration_data:
        pitches
    

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
    duration_model = LinearRegression(X, y_duration, ridge_alpha=0.005)

    with open("out/prediction.txt", "w") as f:
        # TODO: Check this for loop, I don't understand it
        for x in raw_data:
            f.write(f"{x}\n")
        # for n in duration_data:
        #     pitch = VoiceData.get_pitch_from_absolute(n[0])
        #     # print(pitch, n[5])

        #     fileindex = 0
        #     while fileindex < n[preceding_notes]:
        #         f.write(str(pitch))
        #         f.write("\n")
        #         fileindex = fileindex + 1

        idx = 0
        previous_pitch = -1
        durations = [0] * 16
        pitches = [0] * (max_note - min_note + 2)
        while idx < 1500:
            # Perform the prediction
            input = flatten_list(duration_data[-preceding_notes - 1: -1])
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

            # print(f"Predicted pitch: {predicted_pitch}\tduration: {predicted_duration}")

            # Append the encoded predicted pitch and duration to the data to use it as input for the next prediction
            duration_data.append(VoiceData.encode_from_absolute_pitch(predicted_pitch) + [predicted_duration])

            for _ in range(predicted_duration):
                f.write(str(predicted_pitch))
                f.write("\n")

            idx += predicted_duration + 1
    
    pitches_original = [x/sum(pitches_original) for x in pitches_original]
    pitches = [x/sum(pitches) for x in pitches]
    for x in pitches_original:
        print(f"{x: .2f} ", end="")
    print()
    for x in pitches:
        print(f"{x: .2f} ", end="")
    print()
    return distance.euclidean(pitches_original, pitches)



if __name__ == '__main__':
    VOICE = 0
    INCLUDED_PRECEDING_STEPS = range(2, 501, 2)
    d = VoiceData()

    max_cos = 100
    max_idx = -1
    for x in tqdm(INCLUDED_PRECEDING_STEPS): 
        result = main(d, VOICE, x)
        print(f"{x}: {result}")
        if result < max_cos:
            max_cos = result
            max_idx = x
    
    print(f"Max cos: {max_cos} at {max_idx}")