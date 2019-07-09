import argparse
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.merge import _Merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
# from keras.activations import Relu
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
from functools import partial
import cv2

model = Sequential()
model.add(Dense(1024, input_dim=100))
model.add(LeakyReLU())
model.add(Dense(128 * 20 * 15))
model.add(BatchNormalization())
model.add(LeakyReLU())
if K.image_data_format() == 'channels_first':
    model.add(Reshape((128, 20, 15), input_shape=(128 * 15 * 20,)))
    bn_axis = 1
else:
    model.add(Reshape((15, 20, 128), input_shape=(128 * 15 * 20,)))
    bn_axis = -1
model.add(Conv2DTranspose(128, (5, 5), strides=2, padding='same'))
model.add(BatchNormalization(axis=bn_axis))
model.add(LeakyReLU())
model.add(Convolution2D(64, (5, 5), padding='same'))
model.add(BatchNormalization(axis=bn_axis))
model.add(LeakyReLU())
model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
model.add(BatchNormalization(axis=bn_axis))
model.add(LeakyReLU())
# Because we normalized training inputs to lie in the range [-1, 1],
# the tanh function should be used for the output of the generator to ensure
# its output also lies in this range.
model.add(Convolution2D(1, (5, 5), padding='same', activation='tanh'))
model.summary()