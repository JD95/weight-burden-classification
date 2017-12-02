import numpy as np

from keras.models import Sequential, Model, save_model, load_model
from keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D, UpSampling2D, TimeDistributed
from keras.utils import np_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
import re


def parse_data_file(filepath):
    """ String -> [[Float]] """
    data = []
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines[1:]:
            try:
                data.append(list(map(float, line.strip().split())))
            except ValueError:
                continue
    return data


def apply_to_dimensions(f, data):
    """ Apply function to values in every third element """
    return (f(data[0::3]), f(data[1::3]), f(data[2::3]))


def normalize_data(data):
    """ Normalize values in each clip to 0-1 """
    for clip in data:

        current_max = [0, 0, 0]
        current_min = [0, 0, 0]

        for frame in clip:
            # Find max value in that dimension
            d_max = apply_to_dimensions(max, frame)
            for i in range(0, 3):
                current_max[i] = max(d_max[i], current_max[i])

            # Find smallest value in that dimension
            d_min = apply_to_dimensions(min, frame)
            for i in range(0, 3):
                current_min[i] = max(d_min[i], current_min[i])

        for frame in clip:
            for i in range(len(frame)):
                offset = abs(current_min[i % 3])
                scaling = 1.0 / (offset + current_max[i % 3])
                frame[i] = (frame[i] + offset) * scaling
    return data


class DataSet ():
    def __init__(self):
        self.train = []
        self.train_labels = []
        self.test = []
        self.test_labels = []


def add_to_data_set(filepath, category, data_set):
    """ loads a file into the dataset """
    data = parse_data_file(filepath)
    try:
        # Only take the last 5 seconds of the
        # motion to remove the inital readying
        # of the animation
        data = data[len(data) - 600:]
        print(filepath + ' had ' + str(len(data)) + ' frames')
        data_set.train.append(data)
        data_set.train_labels.append(category)
    except IndexError:
        print('The clip from ' + filepath + ' was too short')


def load_data(folder):
    """ loads data for autoencoder """
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    data_set = DataSet()

    for f in files:
        filepath = folder + '/' + f
        if 'light' in f:
            add_to_data_set(filepath, [1.0, 0.0, 0.0], data_set)
        elif 'normal' in f:
            add_to_data_set(filepath, [0.0, 1.0, 0.0], data_set)
        elif 'heavy' in f:
            add_to_data_set(filepath, [0.0, 0.0, 1.0], data_set)
        else:
            pass

    print('loaded ' + str(len(data_set.train)) + ' clips')
    return data_set


def train_auto_encoder(data_set):

    # Setup main training model
    input_layer = Input(shape=(78,))
    encoded = Dense(20, activation='relu')(input_layer)
    decoded = Dense(78, activation='sigmoid')(encoded)
    model = Model(input_layer, decoded)

    # Encoder
    encoder = Model(input_layer, encoded)

    # Decoder
    # encoded_input = Input(shape=(20,))
    # decoder_layer = model.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Compile Models
    model.compile(loss='mean_absolute_error',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(data_set.train, data_set.train, batch_size=256, shuffle=True,
              validation_split=0.2, epochs=150, verbose=1)

    score = model.evaluate(data_set.train, data_set.train, verbose=0)

    print('result is: ' + str(score))

    return encoder


def main():

    data_set = load_data('data')

    print(str(len(data_set.train)) + ' ' + str(len(data_set.train[0])))
    data_set.train = normalize_data(data_set.train)

    print(str(len(data_set.train)) + ' ' + str(len(data_set.train[0])))
    flatten = lambda l: [item for sublist in l for item in sublist]
    data_set.train = flatten(data_set.train) 

    print(str(len(data_set.train)) + ' ' + str(len(data_set.train[0])))
    encoder = None

    try:
        print('Loading encoder...')
        encoder = load_model('encoder.h5')
    except OSError:
        print('Encoder not found, training a new one...')
        encoder = train_auto_encoder(data_set)
        print('Saving encoder to \"encoder.h5\"')
        encoder.save('encoder.h5')

    print(str(len(data_set.train)) + ' ' + str(len(data_set.train[0])))
    # Completely Flatten List for reshaping
    data_set.train = flatten(data_set.train) 

    data_set.train = np.reshape(data_set.train, (191, 600, 78))

    model = Sequential()
    model.add(TimeDistributed(encoder,input_shape=(600,78)))
    model.add(LSTM(4, input_shape=(10, 20)))
    model.add(Dense(3))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(data_set.train, data_set.train_labels,
              epochs=100, batch_size=1, verbose=2)


if __name__ == "__main__":
    main()
