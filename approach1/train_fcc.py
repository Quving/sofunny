from __future__ import print_function

import keras.optimizers as Optimizers
from keras.layers import Dense
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
    embed_dim = 25
    batch_size = 32
    epochs = 25
    learning_rate = 0.001

    # Model
    model = Sequential()
    model.add(Dense(embed_dim, input_dim=embed_dim, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = Optimizers.Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    print_dataset_properties(x_train, y_train)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test loss:', loss)
    print('Test accuracy:', acc)
    export_model(model=model, modelname=modelname)


if __name__ == '__main__':
    modelname = 'fcc_v1'
    x_train, y_train, x_test, y_test = get_dataset_for_lstm(remove_stopwords=True)
    print(len(x_train), len(y_train))
    train_lstm_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
