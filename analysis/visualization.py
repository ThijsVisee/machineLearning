import os
import matplotlib.pyplot as plt
import numpy as np

'''
create a simple plot of the notes of a single voice
'''
def visualize_single_voice(voice):

    plt.plot(voice)
    plt.show()