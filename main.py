import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Convolution2D, MaxPool2D
from keras.utils import np_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

train = []

with open('curl_leftlight_000Parsed.txt') as f:
    lines = f.readlines()
    for line in lines[1:]:
        try:
            train.append(list(map(float, line.strip().split())))
        except ValueError:
            continue

# Setup main training model
input_layer = Input(shape=(78,))
encoded = Dense(20, activation='relu')(input_layer)
decoded = Dense(78, activation='linear')(encoded)
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

model.fit(train, train, batch_size=256, shuffle=True,
          validation_split=0.2, epochs=100, verbose=1)


score = model.evaluate(train, train, verbose=0)

encoded_values = encoder.predict(train)
outputs = decoder.predict(encoded_values)

print('result is: ' + str(score))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train[0][0::3], train[0][1::3], train[0][2::3], c='r')
ax.scatter(outputs[0][0::3], outputs[0][1::3], outputs[0][2::3], c='b')

plt.show()
