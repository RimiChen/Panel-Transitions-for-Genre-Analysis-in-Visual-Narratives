from keras import layers
from keras import models
from keras import __version__ as used_keras_version
import numpy as np


model = models.Sequential()
model.add(layers.Dense(5, activation='sigmoid', input_shape=(1,)))
model.add(layers.Dense(1, activation='sigmoid'))
print((model.predict(np.random.rand(10))))
print('Keras version used: {}'.format(used_keras_version))