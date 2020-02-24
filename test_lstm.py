import numpy as np

from utils.util_model import import_model


def test_lstm():
    modelname = 'lstm_v1'
    model = import_model(modelname=modelname)
    print(model.predict(np.asarray([[12, 41, 412]])))


if __name__ == '__main__':
    test_lstm()
