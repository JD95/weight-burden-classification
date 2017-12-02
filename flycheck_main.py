import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
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
    return (f(data[0::3]), f(data[1::3]), f(data[2::3]))

def normalize_data(data):
    current_max = [0,0,0]
    current_min = [0,0,0]
    for d in data:
        d_max = apply_to_dimensions(max, d)
        for i in range(0,3):
            current_max[i] = max(d_max[i], current_max[i])
        d_min = apply_to_dimensions(min, d)
        for i in range(0,3):
            current_min[i] = max(d_min[i], current_min[i])
    for d in data:
        for i in range(len(d)):
            offset = abs(current_min[i%3])
            scaling = 1.0 / (offset + current_max[i%3])
            d[i] = (d[i] + offset) * scaling
    return data

class DataSet ():
    def __init__(self):
        self.train = []
        self.test = []


def add_to_data_set(filepath, category, data_set):
    """ loads a file into the dataset """
    data = parse_data_file(filepath)
    data_set.train.extend(data)
    data_set.test.extend([category] * len(data))


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

    print('loaded ' + str(len(data_set.train)) + ' samples')
    return data_set


def main():

    data_set = load_data('data')

    data_set.train = normalize_data(data_set.train)

    # Setup main training model
    input_layer = Input(shape=(78,))
    encoded = Dense(20, activation='relu')(input_layer)
    decoded = Dense(78, activation='sigmoid')(encoded)
    model = Model(input_layer, decoded)

    # Encoder
    encoder = Model(input_layer, encoded)

    # Decoder
    encoded_input = Input(shape=(20,))
    decoder_layer = model.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    # Compile Models
    model.compile(loss='mean_absolute_error',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(data_set.train, data_set.train, batch_size=256, shuffle=True,
              validation_split=0.2, epochs=150, verbose=1)

    score = model.evaluate(data_set.train, data_set.train, verbose=0)

    encoded_values = encoder.predict(data_set.train)
    outputs = decoder.predict(encoded_values)

    print('result is: ' + str(score))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # for frame in [(0, 'r'), (1, 'b'), (2, 'g')]:
    #     ax.scatter(outputs[frame[0]][0::3], outputs[frame[0]]
    #                [1::3], outputs[frame[0]][2::3], c=frame[1])

    ax.scatter(data_set.train[0][0::3], data_set.train[0]
               [1::3], data_set.train[0][2::3], c='r')
    ax.scatter(outputs[0][0::3], outputs[0][1::3], outputs[0][2::3], c='b')

    plt.show()


if __name__ == "__main__":
    main()
