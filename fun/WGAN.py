import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU

def generator(data: pd.DataFrame):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(data.shape[0], 1)))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(LSTM(64))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(252))

    return model


def discriminator():
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(252, 1)))  ## (252, 1)
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(LSTM(100))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

