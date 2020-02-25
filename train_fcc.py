from __future__ import print_function

import keras.optimizers as Optimizers
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from utils.util_dataset import get_dataset_for_fcc
from utils.util_model import export_model


def train_fcc_model(x_train, y_train, x_test, y_test):
    # Parameters
    batch_size = 32
    epochs = 100

    # Model
    model = Sequential()
    model.add(Dense(34, input_dim=34, activation='tanh'))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(200, activation="tanh"))
    model.add(Dense(100, activation="tanh"))
    model.add(Dense(1, activation="tanh"))

    opt = Optimizers.RMSprop(lr=0.01)
    model.compile(loss='mse',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)
    export_model(model=model, modelname='fcc_v1')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_dataset_for_fcc()
    print("===============================")
    print(len(x_train[0]))
    print(len(x_train), len(y_train))
    train_fcc_model(
        x_train=np.asarray(x_train),
        y_train=y_train,
        x_test=np.asarray(x_test),
        y_test=y_test
    )
