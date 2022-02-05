import numpy as np
from data.data_loader import VoiceData 

'''
print and return a few basic statistics about a single voice
'''
def get_voice_statistics(voice, include_zeroes = True, silent = False):

    arr = voice
    # check if data is encoded or not
    if(isinstance(arr[0], (list, np.ndarray))):
        arr = VoiceData.get_voice_from_encoding(arr)

    stats = {}
    if(not include_zeroes):
        arr = arr[np.nonzero(arr)]

    stats['mean'] = np.mean(arr)
    stats['std'] = np.std(arr)
    stats['max'] = np.amax(arr)
    stats['min'] = np.amin(arr)

    if(not silent):
        print(f"Mean: {stats['mean']}; Standard Deviation: {stats['std']}; Min: {stats['min']}; Max: {stats['max']}")

    return stats

'''
return a few basic statistics about all voices
'''
def get_statistics(data, include_zeroes = True, silent = False):

    stats = []
    for idx, d in enumerate(data):
        substats = (get_voice_statistics(d, include_zeroes, silent))
        substats['voice'] = idx
        stats.append(substats)
    
    return stats