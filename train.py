import csv
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def get_trainingdata():
    with open("data/task-1/train.csv", "r") as file:
        csv_reader = csv.reader(file, delimiter=",", )
        next(csv_reader)
        return list(csv_reader)


def preprocess_data(data):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    def extract_woi(sentence):
        p = re.compile("<(.*)/>")
        result = p.search(sentence)
        sentence = sentence.replace(result.group(0), "xxxxx")
        return sentence.lower(), result.group(1).lower()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    x_train, y_train = [], []
    for data_point in data:
        original_raw = data_point[1]
        original, woi = extract_woi(original_raw)

        # Filter only alphanumeric and whitespace.
        original = re.sub(r'\W+', ' ', original.strip().lower())

        # Remove stopwords and apply stemming/lemmatizing
        word_tokens = word_tokenize(original)
        filtered_sentence = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words]

        print(original)
        print(filtered_sentence)
        print(woi)


if __name__ == '__main__':
    data = get_trainingdata()
    data = preprocess_data(data[:5])
