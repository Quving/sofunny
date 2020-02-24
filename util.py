import csv
import re

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

placeholder = 'xxxxx'
num_words = 10000


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
    return x_data[:index], y_data[:index], x_data[index:], y_data[index:]


def get_trainingdata(filename):
    """
    Load local training data from csv.
    Returns:

    """
    with open(filename, "r") as file:
        csv_reader = csv.reader(file, delimiter=",", )
        next(csv_reader)
        return list(csv_reader)


def convert_data_to_basic(data):
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
        filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words]
        sentences.append(" ".join(filtered_sentence))
        woi1.append(woi)
        woi2.append(data_point[2])
        grades.append(data_point[4])

    return sentences, woi1, woi2, grades


def convert_to_trainingdata_for_lstm(sentences, woi1, woi2, grades):
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
        sentences_train.append(s.replace(placeholder, w1))
        sentences_train.append(s.replace(placeholder, w2))
        grades_train.append([0.0])
        grades_train.append([float(g)])

    # Tokenize the training sentences according to its frequency.
    tokenizer = Tokenizer(num_words=num_words, split=' ')
    tokenizer.fit_on_texts(sentences_train)
    # print(tokenizer.word_index)  # To see the dicstionary
    # print(tokenizer.document_count)  # To see the dicstionary
    x_train = tokenizer.texts_to_sequences(sentences_train)

    # Left pad the training sequences.
    x_train = pad_sequences(x_train)
    y_train = np.asarray(grades_train)

    assert len(x_train) == len(y_train)
    return x_train, y_train


def get_dataset_for_lstm():
    trainset = 'data/task-1/train.csv'
    validationset = 'data/task-1/dev.csv'

    # Trainingset
    dataset_train = get_trainingdata(filename=trainset)
    dataset_val = get_trainingdata(filename=validationset)
    dataset = dataset_train + dataset_val
    sentences, woi1, woi2, grades = convert_data_to_basic(dataset)
    x_train, y_train = convert_to_trainingdata_for_lstm(sentences=sentences,
                                                        woi1=woi1,
                                                        woi2=woi2,
                                                        grades=grades)

    return split_dataset(x_data=x_train, y_data=y_train, ratio=0.8)
