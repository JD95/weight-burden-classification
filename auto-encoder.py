from keras.layer import Input, Dense
from keras.models import Model

input_dim = 111 
encoding_dim = 20

input_frame = None

encoded = Dense(encoding_dim, activation='relu')(input_frame)

decoded = Dense(input_dim, activation='relu')(encoded)

autoencoder = Model(input_frame, decoded)

encoder = Model(input_frame, encoded)

encoded_input = Input(shape=(encoding_dim))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
