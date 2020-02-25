from util_dataset import load_raw_dataset, convert_data_to_basic, convert_to_trainingdata
from util_model import import_model


def eval_lstm():
    # Prepare test dataset
    trainset = 'data/task-1/test.csv'

    # Trainingset
    dataset_train = load_raw_dataset(filename=trainset)
    dataset = dataset_train
    sentences, woi1, woi2, grades = convert_data_to_basic(dataset, remove_stopwords=False)
    x_train, y_train = convert_to_trainingdata(sentences=sentences,
                                               woi1=woi1,
                                               woi2=woi2,
                                               grades=grades,
                                               use_stored_tokenizer=True)

    # Load model
    modelname = 'fcc_v1'
    model = import_model(modelname=modelname)
    predictions = model.predict(x_train)

    csv_dict = {"id": [], "pred": []}

    for index, prediction in enumerate(predictions):
        prediction = list(prediction)
        max_val = max(prediction)
        classname = prediction.index(max_val) / 10
        id = dataset_train[index][0]
        print(id, classname)

        csv_dict["id"].append(index)
        csv_dict["pred"].append(classname)


if __name__ == '__main__':
    eval_lstm()
