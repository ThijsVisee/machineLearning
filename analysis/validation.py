import math
import os
import numpy as np
import matplotlib.pyplot as plt

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
print a simple plot of the notes
'''
def visualize_results(filepath):
    fTitle = f'{os.getcwd()}/out/{filepath}.txt'
    file = np.loadtxt(fTitle, dtype=int)
    plt.plot(file)
    plt.show()