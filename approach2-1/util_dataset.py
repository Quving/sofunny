import csv
import pickle
import re

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

placeholder = 'xxxxx'
num_words = 10000  # Max features
max_grade = 3
max_embedded_size = 25


def split_dataset(x_data, y_data, ratio):
    """
    Split a given dataset into a training and validation set with a given ratio.
    Args:
        x_data:
        y_data:
        ratio: 0.0 - 1.0 indicates the percentage of trainingset that should be kept.
        0.8 means 80% trainingset and 20 validation set.

    Returns:
    """
    assert ratio <= 1.0
    index = int(len(x_data) * ratio)
    return np.asarray(x_data[:index]), \
           np.asarray(y_data[:index]), \
           np.asarray(x_data[index:]), \
           np.asarray(y_data[index:])


def load_raw_dataset(filename):
    """
    Load local training data from csv.
    Returns:

    """
    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=",", )
        next(csv_reader)
        return list(csv_reader)


def convert_data_to_basic(data, remove_stopwords):
    """
    Convert the dataset into a basic form. Using 'xxxxx' as placeholder for the word of interest.
    Args:
        data:

    Returns:
        sentences, woi1, woi2, grades
    """

    def extract_woi(sentence):
        p = re.compile("<(.*)/>")
        result = p.search(sentence)
        sentence = sentence.replace(result.group(0), placeholder)
        return sentence.lower(), result.group(1).lower()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    sentences, woi1, woi2, grades = [], [], [], []
    for data_point in data:
        original_raw = data_point[1]
        original, woi = extract_woi(original_raw)

        # Filter only alphanumeric and whitespace.
        original = re.sub(r'\W+', ' ', original.strip().lower())

        # Remove stopwords and apply stemming/lemmatizing
        word_tokens = word_tokenize(original)
        if remove_stopwords:
            filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words]
        else:
            filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens]

        sentences.append(" ".join(filtered_sentence))
        woi1.append(woi)
        woi2.append(data_point[2])

        # If statement for test dataset that does not contain grade column.
        if len(data_point) == 5:
            grades.append(data_point[4])
        else:
            grades.append(-1.0)

    return sentences, woi1, woi2, grades


def convert_to_trainingdata(sentences, woi1, woi2, grades, use_stored_tokenizer=False):
    """
    Converts to a trainable dataformat for lstms where x is an array of indices and y is the grade.
    Args:
        sentences:
        woi1:
        woi2:
        grades:

    Returns:

    """

    # Fill placeholders in sentences
    sentences_train = []
    grades_train = []
    for s, w1, w2, g in zip(sentences, woi1, woi2, grades):
        # sentences_train.append(s.replace(placeholder, w1))
        sentences_train.append(s.replace(placeholder, w2))
        # grades_train.append([0.0])
        grades_train.append([float(g)])

    if not use_stored_tokenizer:
        tokenizer = Tokenizer(num_words=num_words, split=' ')
        tokenizer.fit_on_texts(sentences_train)
        with open('models/tokenizer.pickle', 'wb') as pickle_file:
            pickle.dump(tokenizer, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Load local tokenizer from file")
        with open('models/tokenizer.pickle', 'rb') as pickle_file:
            tokenizer = pickle.load(pickle_file)

    # print(tokenizer.word_index)  # To see the dicstionary
    # print(tokenizer.document_count)  # To see the dicstionary
    x_train = tokenizer.texts_to_sequences(sentences_train)

    # Left pad the training sequences.
    x_train = pad_sequences(x_train, maxlen=max_embedded_size)
    y_train = np.asarray(grades_train)

    assert len(x_train) == len(y_train)

    # Normalize only x_train
    x_train, y_train = normalize_data(x_train, y_train, num_words, 1)
    return np.asanyarray(x_train), np.asanyarray(y_train)


def normalize_data(x_train, y_train, x_train_max, y_train_max):
    def normalize_datapoint(d, n_max):
        return list(map(lambda n: n / n_max, d))

    x_train_normalized = list(map(lambda x: normalize_datapoint(x, x_train_max), x_train))
    y_train_normalized = list(map(lambda x: normalize_datapoint(x, y_train_max), y_train))

    return x_train_normalized, y_train_normalized


def denormalize_data(x_train, y_train, x_train_max, y_train_max):
    def denormalize_datapoint(d, n_max):
        return list(map(lambda n: n * n_max, d))

    x_train_normalized = list(map(lambda x: denormalize_datapoint(x, x_train_max), x_train))
    y_train_normalized = list(map(lambda x: denormalize_datapoint(x, y_train_max), y_train))

    return x_train_normalized, y_train_normalized


def get_dataset_for_lstm(remove_stopwords=False):
    trainset = 'data/task-1/train.csv'
    validationset = 'data/task-1/dev.csv'

    # Trainingset
    dataset_train = load_raw_dataset(filename=trainset)
    dataset_val = load_raw_dataset(filename=validationset)
    dataset = dataset_train + dataset_val
    sentences, woi1, woi2, grades = convert_data_to_basic(dataset, remove_stopwords=remove_stopwords)
    x_train, y_train = convert_to_trainingdata(sentences=sentences,
                                               woi1=woi1,
                                               woi2=woi2,
                                               grades=grades)

    return split_dataset(x_data=x_train, y_data=y_train, ratio=0.8)
