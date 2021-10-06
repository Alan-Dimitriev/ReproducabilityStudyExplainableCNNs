import csv
from torchtext.data import get_tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer


def reader(file):
    with open(file, "r") as f:
        csvreader = csv.reader(f)
        comments = []
        labels = []
        for rows in csvreader:
            comments.append(rows[1])
            labels.append(rows[2:])
    return comments, labels


def modelling(input_, output):
    # 20% test samples, random state set for reproducability across multiple runs
    X_train, X_test, y_train, y_test = train_test_split(input_, output, test_size=0.2,
                                                        random_state=42)
    tokenizer = get_tokenizer("basic_english")  # todo make sure this is the token we want
    tokens = tokenizer(X_train[0])
    print(tokens)


comments, labels = reader("data/train.csv")
modelling(comments, labels)
