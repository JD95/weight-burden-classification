import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D
from keras.utils import np_utils

train = []

with open('curl_leftlight_000Parsed.txt') as f:
    lines = f.readlines()
    for line in lines[1:]:
        try:
            train.append(list(map(float, line.strip().split())))
        except ValueError:
            continue

model = Sequential()

model.add(Dense(20, input_dim=78, activation='relu'))
model.add(Dense(78, activation='relu'))

model.compile(loss='mean_squared_error',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(train, train, batch_size=256, shuffle=True,
          validation_split=0.2, epochs=1000, verbose=1)

score = model.evaluate(train, train, verbose=0)

print('result is: ' + str(score))
