import math
import numpy as np


'''
calculate the loss
'''
def calculate_loss():
    return

'''
return the mean squared logarithmic error (MSLE).
Both input lists need to have the same size
'''
def msle(hyp, val):
    if len(hyp) != len(val):
        raise RuntimeError(f"length {len(hyp)} != {len(val)}")
    # we add +1 to both values here to avoid stupid errors in case one of the values is zero
    err = map(lambda x, y: math.pow(math.log10(x+1) - math.log10(y+1),2), hyp, val)

    return np.mean(list(err))

'''
return the difference between the mean of the prediction and the ground truth
'''
def mean_error(pred, grt):
    if(not isinstance(pred, (np.ndarray))):
        pred = np.array(pred)
    
    if(not isinstance(grt, (np.ndarray))):
        grt = np.array(grt)

    return abs(np.mean(pred) - np.mean(grt))


'''
return the difference between the standard deviation of the prediction and the ground truth
'''
def std_error(pred, grt):
    if(not isinstance(pred, (np.ndarray))):
        pred = np.array(pred)
    
    if(not isinstance(grt, (np.ndarray))):
        grt = np.array(grt)

    return abs(np.std(pred) - np.std(grt))