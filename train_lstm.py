from __future__ import print_function

import keras.optimizers as Optimizers
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential

from util_dataset import get_dataset_for_lstm
from util_model import export_model


def print_dataset_properties(x_train, y_train):
    max_index, min_index = 0, 0
    for x in x_train:
        max_index = max(x) if max(x) > max_index else max_index
        min_index = min(x) if min(x) > min_index else min_index

    print("Max index {} ".format(max_index))
    print("Min index {} ".format(min_index))
    print("Max grade {} ".format(max(y_train)))
    print("Min grade {} ".format(min(y_train)))


def train_lstm_model(x_train, y_train, x_test, y_test):
    # Parameters
    embed_dim = len(x_train[0])
    batch_size = 32
    epochs = 10
    max_features = 2000

    # Model
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, dropout=0.1))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(200, activation='tanh'))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    opt = Optimizers.RMSprop(lr=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    print_dataset_properties(x_train, y_train)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)
    export_model(model=model, modelname='lstm_v1')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_dataset_for_lstm()
    print(len(x_train), len(y_train))
    train_lstm_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
