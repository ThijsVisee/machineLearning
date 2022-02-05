import os
import matplotlib.pyplot as plt
import numpy as np
from data.data_loader import VoiceData 


PLOTDIR = 'out/plots'

'''
create a simple plot of the notes of a single voice
'''
def plot_single_voice(voice, vIdx, centered = True):

    check_dir_exists(PLOTDIR)

    fname = f'{PLOTDIR}/voice{vIdx+1}.png'

    arr = voice
    # check if data is encoded or not
    if(isinstance(arr[0], (list, np.ndarray))):
        arr = VoiceData.get_voice_from_encoding(arr)

    if(centered):
        mean = np.mean(arr)
        
        with np.nditer(arr, op_flags=['readwrite']) as it:
            for val in it:
                if val == 0:
                    val[...] = mean

        fname = f'{PLOTDIR}/voice{vIdx+1}_centered.png'

    plt.figure()
    plt.plot(arr)
    plt.savefig(fname, format='png', dpi=600, transparent=True)


'''
create a combined plot of all voices
'''
def plot_all_voices(data, centered = True):

    check_dir_exists(PLOTDIR)

    plt.figure()

    fname = f'{PLOTDIR}/all_voices.png'

    if centered:
        fname = f'{PLOTDIR}/all_voices_centered.png'

    for voice in data:

        arr = voice
        # check if data is encoded or not
        if(isinstance(arr[0], (list, np.ndarray))):
            arr = VoiceData.get_voice_from_encoding(arr)

        if(centered):
            mean = np.mean(arr)
            
            with np.nditer(arr, op_flags=['readwrite']) as it:
                for val in it:
                    if val == 0:
                        val[...] = mean
    
        plt.plot(arr)

    plt.savefig(fname, format='png', dpi=600, transparent=True)


def check_dir_exists(dir):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)