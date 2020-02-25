from __future__ import print_function

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

from util_dataset import get_dataset_for_lstm
from util_model import export_model


def train_lstm_model(x_train, y_train, x_test, y_test):
    # Parameters
    embed_dim = len(x_train[0])
    batch_size = 32
    epochs = 100
    num_classes = 31

    y_train = list(map(lambda n: n * 10, y_train))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = list(map(lambda n: n * 10, y_test))
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=embed_dim))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', input_dim=embed_dim))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.metrics_names)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Test loss:', loss)
    print('Test accuracy:', acc)
    export_model(model=model, modelname=modelname)


if __name__ == '__main__':
    modelname = 'fcc_v1'
    x_train, y_train, x_test, y_test = get_dataset_for_lstm(remove_stopwords=True)
    print(len(x_train) + len(x_test))
    train_lstm_model(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )
