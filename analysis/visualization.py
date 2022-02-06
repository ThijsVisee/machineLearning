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

    plt.figure(figsize=(14.4, 4.8))
    plt.plot(arr)

    plt.xlabel("Time Steps")
    plt.ylabel("Piano Key Index")

    plt.savefig(fname, format='png', dpi=600, transparent=True)
    plt.close()


'''
create a combined plot of all voices
'''
def plot_all_voices(data, centered = True):

    check_dir_exists(PLOTDIR)

    plt.figure(figsize=(14.4, 4.8))

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
    
    plt.xlabel("Time Steps")
    plt.ylabel("Piano Key Index")
    plt.legend(["Voice 1","Voice 2","Voice 3","Voice 4"])  

    plt.savefig(fname, format='png', dpi=600, transparent=True)

    plt.close()

def plot_error_rate(measure, unit, title, data):

    check_dir_exists(PLOTDIR)

    plt.figure()

    fname = f'{PLOTDIR}/{title}.png'

    plt.plot(data)

    plt.xlabel(unit)
    plt.ylabel(measure)

    plt.savefig(fname, format='png', dpi=600, transparent=True)

    plt.close()


def boxplot(data, centered = True):

    plots = []

    fname = f'{PLOTDIR}/boxplot.png'

    for d in data:
        arr = d
        if(isinstance(d[0], (list, np.ndarray))):
            arr = VoiceData.get_voice_from_encoding(d)
        
        if(centered):
            mean = np.mean(arr)
            
            with np.nditer(arr, op_flags=['readwrite']) as it:
                for val in it:
                    if val == 0:
                        val[...] = mean

            fname = f'{PLOTDIR}/boxplot_centered.png'

        plots.append(arr)

    check_dir_exists(PLOTDIR)

    plt.figure()

    plt.boxplot(plots)

    plt.xlabel("Voices")
    plt.ylabel("Piano Key Index")

    plt.savefig(fname, format='png', dpi=600, transparent=True)


def check_dir_exists(dir):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)