import numpy as np

import keras
from keras.models import Sequential, Model, save_model, load_model
from keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D, UpSampling2D, TimeDistributed
from keras.utils import np_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, join
import re
import random


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
            add_to_data_set(filepath, np.array([1.0, 0.0, 0.0]), data_set)
        elif 'normal' in f:
            add_to_data_set(filepath, np.array([0.0, 1.0, 0.0]), data_set)
        elif 'heavy' in f:
            add_to_data_set(filepath, np.array([0.0, 0.0, 1.0]), data_set)
        else:
            pass

    print('loaded ' + str(len(data_set.train)) + ' clips')
    return data_set


def y_rotation_matrix(theta):
    return np.matrix([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]])


def data_aug(data, labels):
    """
    data is 189x600x78

    taking 360 length slice (3 seconds), giving 240 subsamples

    rotating subsamples 359 times around y axis
    """
    aug_data = np.zeros(90 * 360 * 78)
    aug_data = np.reshape(aug_data, (90, 360, 78))
    point = np.zeros(3)
    batch_size = 90

    while True:
        for s in range(0, len(data)):
            s_labels = np.reshape(np.repeat(labels[s],batch_size), (batch_size, 3))
            for b in range(0,batch_size):
                    
                subsample = random.randint(0,240)
                theta = random.randint(0,360)
                rot = y_rotation_matrix(theta)

                # Take 3 second subsample
                for i in range(subsample, subsample + 359):

                    # forall subsamples
                    for j in range(0, 26):
                        j0 = 3*j
                        j1 = j0 + 1
                        j2 = j0 + 2
                        point[0] = data[s][i][j0]
                        point[1] = data[s][i][j1]
                        point[2] = data[s][i][j2]
                        point = (point * rot).tolist()[0]
                        i_sample = i - subsample
                        aug_data[b][i_sample][j0] = point[0]
                        aug_data[b][i_sample][j1] = point[1]
                        aug_data[b][i_sample][j2] = point[2]

            yield (aug_data, s_labels)


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


def train_LSTM(encoder, data_set):
    model = Sequential()
    model.add(TimeDistributed(encoder, input_shape=(360, 78), name="TimeDense"))
    model.add(LSTM(20, input_shape=(10, 20), dropout=0.25, name="LSTM"))
    model.add(Dense(3, activation="softmax", name="DenseLayer"))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    tbCallback = keras.callbacks.TensorBoard(log_dir='./Graph',
                                             histogram_freq=0,
                                             write_graph=True,
                                             write_images=True)
    model.fit_generator(data_aug(data_set.train, data_set.train_labels),
                        epochs=100,
                        steps_per_epoch=189,
                        verbose=2,
                        callbacks=[tbCallback])

    return model


def main():

    print("Loading Data...")
    data_set = load_data('data')

    print("Normalizing Data...")
    data_set.train = normalize_data(data_set.train)

    def flatten(l): return [item for sublist in l for item in sublist]
    data_set.train = flatten(data_set.train)

    encoder = None

    try:
        print('Loading encoder...')
        encoder = load_model('encoder.h5')
    except OSError:
        print('Encoder not found, training a new one...')
        encoder = train_auto_encoder(data_set)
        print('Saving encoder to \"encoder.h5\"')
        encoder.save('encoder.h5')

    # Completely Flatten List for reshaping
    data_set.train = flatten(data_set.train)

    data_set.train = np.reshape(data_set.train, (189, 600, 78))

    data_set.test = data_set.train[0:37]
    data_set.test_labels = data_set.test[0:37]

    data_set.train = data_set.train[37:]
    data_set.train_labels = data_set.train_labels[37:]

    try:
        print('Loading LSTM...')
        model = load_model('classifer.h5')
    except OSError:
        print('Classifier not found, training a new one...')
        model = train_LSTM(encoder, data_set)
        print('Saving classifer to \"classifer.h5\"')
        model.save('classifer.h5')

    score = model.evaluate(data_set.test, data_set.test_labels, verbose=0)
    print('score is: ' + str(score))


if __name__ == "__main__":
    main()
