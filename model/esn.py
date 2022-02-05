

# import tensorflow addons
import tensorflow_addons as tfa
from tensorflow import keras 
import numpy as np

inputs = np.random.random([30,23,9]).astype(np.float32)
ESNCell = tfa.rnn.ESNCell(4)
rnn = keras.layers.RNN(ESNCell, return_sequences=True, return_state=True)
outputs, memory_state = rnn(inputs)
# outputs.shape

# memory_state.shape

