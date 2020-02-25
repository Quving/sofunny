from util_dataset import load_raw_dataset, convert_data_to_basic, convert_to_trainingdata_for_lstm, max_grade
from util_model import import_model


def test_lstm():
    # Prepare test dataset
    trainset = 'data/task-1/test.csv'

    # Trainingset
    dataset_train = load_raw_dataset(filename=trainset)
    dataset = dataset_train
    sentences, woi1, woi2, grades = convert_data_to_basic(dataset, remove_stopwords=False)
    x_train, y_train = convert_to_trainingdata_for_lstm(sentences=sentences,
                                                        woi1=woi1,
                                                        woi2=woi2,
                                                        grades=grades,
                                                        use_stored_tokenizer=True)
    # Load model
    modelname = 'lstm_v1'
    model = import_model(modelname=modelname)
    predictions = model.predict(x_train)
    predictions = list(map(lambda n: n * max_grade, predictions))

    print("Max {}".format(max(predictions)))
    print("Min {}".format(min(predictions)))


if __name__ == '__main__':
    test_lstm()
