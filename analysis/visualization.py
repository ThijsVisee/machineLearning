import os
import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import VoiceData 

'''
create a simple plot of the notes of a single voice
'''
PLOTDIR = 'out/plots'

def visualize_single_voice(voice, vIdx, include_zeroes = True):

    check_dir_exists(PLOTDIR)

    arr = voice
    # check if data is encoded or not
    if(isinstance(arr[0], (list, np.ndarray))):
        arr = VoiceData.get_voice_from_encoding(arr)

    if(not include_zeroes):
        arr = arr[np.nonzero(arr)]

    fname = f'{PLOTDIR}/voice{vIdx+1}.png'

    plt.plot(arr)
    plt.savefig(fname, format='png', dpi=600, transparent=True)


def check_dir_exists(dir):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)