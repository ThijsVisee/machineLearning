import math
import numpy as np
from scipy.special import softmax
from data.data_loader import VoiceData

def get_prob_vec(voice, include_zeroes = False):

    arr = voice
    # check if data is encoded or not
    if(isinstance(arr[0], (list, np.ndarray))):
        arr = VoiceData.get_voice_from_encoding(arr)
    
    if(not include_zeroes):
        arr = arr[np.nonzero(arr)]

    all_pitches = [0] * (VoiceData._VoiceData__highest_note - VoiceData._VoiceData__lowest_note + 2)

    for note in arr:
        all_pitches[note] +=1

    return softmax(all_pitches)
    #return [x/sum(all_pitches) for x in all_pitches]

'''
return the mean squared logarithmic error (MSLE).
Both input lists need to have the same size
'''
def msle(hyp, val):
    if len(hyp) != len(val):
        raise RuntimeError(f"length {len(hyp)} != {len(val)}")
    # we add +1 to both values here to avoid stupid errors in case one of the values is zero
    err = map(lambda x, y: math.pow(math.log10(x + 1) - math.log10(y + 1), 2), hyp, val)
    errList = list(err)

    return np.mean(errList)


'''
return the difference between the mean of the prediction and the ground truth
'''


def mean_error(pred, grt):
    if (not isinstance(pred, (np.ndarray))):
        pred = np.array(pred)

    if (not isinstance(grt, (np.ndarray))):
        grt = np.array(grt)

    return abs(np.mean(pred) - np.mean(grt))


'''
return the difference between the standard deviation of the prediction and the ground truth
'''


def std_error(pred, grt):
    if (not isinstance(pred, (np.ndarray))):
        pred = np.array(pred)

    if (not isinstance(grt, (np.ndarray))):
        grt = np.array(grt)

    return abs(np.std(pred) - np.std(grt))
