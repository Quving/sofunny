from __future__ import print_function

import keras.optimizers as Optimizers
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential

from util import get_trainingdata_for_lstm


def train_lstm_model(x_train, y_train, x_test, y_test):
    # Parameters
    embed_dim = len(x_train[0])
    batch_size = 32
    epochs = 50
    max_features = 10000
    num_classes = 40

    # Model
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, dropout=0.1))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    opt = Optimizers.RMSprop(lr=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    x_train, y_train = get_trainingdata_for_lstm()
    x_test, y_test = [], []
    # Split to train and validation set.
    print(x_train[0], y_train[0])
    train_lstm_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
