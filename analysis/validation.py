import math
import numpy as np

'''
calculate the loss
'''
def calculate_loss():
    return

'''
return the mean squared logarithmic error (MSLE)
'''
def msle(hyp, val):
    err = map(lambda x, y: math.pow(math.log10(x+1) - math.log10(y+1)), hyp, val)

    return np.mean(err)


if __name__ == '__main__':
    from data.data_loader import VoiceData
    from main import main
    VOICE = 0
    # values below are multiplied by 16 to get the actual number of notes from bars
    INCLUDED_PRECEDING_STEPS = 12 * 16
    PREDICTION = 24 * 16

    d = VoiceData()

    model = main(d, VOICE, INCLUDED_PRECEDING_STEPS, PREDICTION, False)