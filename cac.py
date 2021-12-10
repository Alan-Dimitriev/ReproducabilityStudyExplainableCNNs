import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import pickle
import math
from ast import literal_eval
import tqdm
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.layers import *
import keras
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPool1D, MaxPool2D
from sklearn.metrics import accuracy_score, classification_report


def get_length(input_list):
    """
    Helper function that returns the length of an input list
    :param input_list: document list to return the length for
    :return: the list's length
    """
    return len(input_list)


def embed_text(input_list, model):
    """
    Helper function to embed a document quickly based upon an embedding model
    :param input_list: the document that needs to be embedded
    :param model: the model itself that will do the embedding.
    :return: a word vector with the embedding values
    """
    return model.wv[input_list]


def flatten(t):
    """
    Helper function to flatten the document of vectors, used to reduce the dimensions of an input
    :param t: the 3D matrix that needs to be made into 2D
    :return: a 2D list which is a flattened version of t.
    """
    return [item for sublist in t for item in sublist]


def text_cnn_build():
    """
    Builds the Text CNN Model
    :return: The Text CNN Model
    """
    data_input = tf.keras.Input(shape=(864, 100))  # 864 features (based on number of words in the truncated document) & each word is 100 val long vector

    conv = Conv1D(500, kernel_size=4, activation='relu',
                  kernel_regularizer=keras.regularizers.l2(3))(data_input)
    pl = MaxPool1D(5)(conv)

    dropout = tf.keras.layers.Dropout(.2)(pl)

    flattened = Flatten()(dropout)

    dense = Dense(50, activation='softmax')(flattened)

    model = keras.Model(inputs=data_input, outputs=dense)
    return model


def text_cnn(x_train, y_train, x_test, y_test):
    """
    Trains and scores the text cnn model using the following parameters which were split from the original MIMICIII Database
    :param x_train: split of feature values to train upon
    :param y_train: split of label values to train against
    :param x_test: split of feature values to test upon and predict with
    :param y_test: split of label values to test with, and score based on the predictions on x_test
    :return: nothing
    """
    print("building text cnn")
    model = text_cnn_build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['AUC'])
    model.summary()
    print("about to fit the model")
    model.fit(x=x_train, y=y_train, batch_size=16, validation_split=.2,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
                                                   restore_best_weights=True)
              ], epochs=100)
    print("model has been trained and fit")
    predictions = model.predict(x_test)
    predictions = np.round(predictions)

    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


def swam_caml_build():
    """
    Builds the SWAM CAML Model
    :return: The SWAM CAML Model
    """
    data_input = tf.keras.Input(shape=(864, 100))  # 864 features (based on number of words in the truncated document) & each word is 100 val long vector

    conv = Conv1D(500, kernel_size=4, activation='softmax')(data_input)  # wide and thus 500 filters
    dropout = tf.keras.layers.Dropout(.2)(conv)

    query_value_attention_seq = tf.keras.layers.Attention()([dropout, dropout])

    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(conv)

    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    # flattened = Flatten()(conv)

    dense = Dense(50, activation='sigmoid')(input_layer)

    model = keras.Model(inputs=data_input, outputs=dense)
    return model


def swam_caml(x_train, y_train, x_test, y_test):
    """
    Trains and scores the SWAM CAML model using the following parameters which were split from the original MIMICIII Database
    :param x_train: split of feature values to train upon
    :param y_train: split of label values to train against
    :param x_test: split of feature values to test upon and predict with
    :param y_test: split of label values to test with, and score based on the predictions on x_test
    :return: nothing
    """
    print("building swam caml")
    model = swam_caml_build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['AUC'])
    model.summary()
    print("about to fit the model")
    model.fit(x=x_train, y=y_train, batch_size=16, validation_split=.2,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
                                                   restore_best_weights=True)
              ], epochs=100)
    print("model has been trained and fit")
    predictions = model.predict(x_test)
    predictions = np.round(predictions)

    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


def caml_build():
    """
     Builds the CAML Model
     :return: The CAML Model
     """
    data_input = tf.keras.Input(shape=(864, 100))  # 864 features (based on number of words in the truncated document) & each word is 100 val long vector

    conv = Conv1D(50, kernel_size=4, activation='softmax')(data_input)  # the main difference is that it is only 50 filters and thus not wide
    dropout = tf.keras.layers.Dropout(.2)(conv)

    query_value_attention_seq = tf.keras.layers.Attention()([dropout, dropout])

    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(conv)

    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    input_layer = tf.keras.layers.Concatenate()(
        [query_encoding, query_value_attention])

    dense = Dense(50, activation='sigmoid')(input_layer)

    model = keras.Model(inputs=data_input, outputs=dense)
    return model


def caml(x_train, y_train, x_test, y_test):
    """
    Trains and scores the CAML model using the following parameters which were split from the original MIMICIII Database
    :param x_train: split of feature values to train upon
    :param y_train: split of label values to train against
    :param x_test: split of feature values to test upon and predict with
    :param y_test: split of label values to test with, and score based on the predictions on x_test
    :return: nothing
    """
    print("building caml")
    model = caml_build()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        loss='categorical_crossentropy',
        metrics=['AUC'])
    model.summary()
    print("about to fit the model")
    model.fit(x=x_train, y=y_train, batch_size=16, validation_split=.2,
              callbacks=[
                  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10,
                                                   restore_best_weights=True)
              ], epochs=100)
    print("model has been trained and fit")
    predictions = model.predict(x_test)
    predictions = np.round(predictions)

    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))


def main():
    """
    all the running of the different models as well as part of the pre-processing happens here
    :return: nothing, but prints the scores of the different models.
    """
    print("it is starting")

    df = pd.read_csv("final_data_hot_encoded.csv")  # read the hot encoded CSV

    df.TEXT = df.TEXT.apply(literal_eval)   # after reading it, evaluate the TEXT as a list of words
    df.CODES = df.CODES.apply(literal_eval)  # after reading it, evaluate the CODES as a list of labels/codes for the documents

    embed = gensim.models.word2vec.Word2Vec.load("first_model.model")  # load the model that was created prior

    df['TEXT'] = df['TEXT'].apply(lambda x: embed_text(x, embed))  # embed the text using the model
    df['LENGTH'] = df['TEXT'].apply(lambda x: get_length(x))  # make a new column with the lengths of each document
    print(df['TEXT'].str.len().agg(['mean', 'max', 'std']))

    #  following code finds the truncated value by finding what max length 90% of documents have.
    text_length = df.LENGTH.to_list()  # make a list of document lengths
    text_length.sort()
    trunc_val = text_length[math.floor(0.9 * len(text_length))]

    # following code will truncate the documents if too long, or pad them if too short.
    num_rows = df.iterrows()
    for index, row in tqdm(num_rows):
        input_list = row['TEXT']
        if len(input_list) >= trunc_val:
            input_list = input_list[:trunc_val]
            df.at[index, 'TEXT'] = input_list
        else:
            itera = math.ceil((trunc_val - len(input_list)) / 2)
            input_list = np.concatenate(
                (np.zeros((itera, 100)), input_list, np.zeros((itera, 100))))
            input_list = input_list[:trunc_val]
            df.at[index, 'TEXT'] = input_list


    # the following code was used to pickle and store the padded/truncated documents and read them later to save time.
    # df.to_pickle('/global/home/hpc4748/Masters/CISC867/Projects/ReproducabilityStudyExplainableCNNs/pre-processed-docs/Padded_DF.pkl')
    # df = pd.read_pickle(
    #    '/global/home/hpc4748/Masters/CISC867/Projects/ReproducabilityStudyExplainableCNNs/pre-processed-docs/Padded_DF.pkl')
    # print("Pkl has been read")

    print(df['TEXT'].str.len().agg(['mean', 'max', 'std']))
    X = df.TEXT.to_list()
    print("csv TEXT has been converted into a list")

    # the following code is for Logistic Regression which only took in 2D data.
    for i in tqdm(range(len(X))):
        X[i] = flatten(X[i])

    invalid = ['TEXT', 'CODES', 'LENGTH']
    codes_df = df[df.columns.difference(invalid)]
    codes_df_list = codes_df.values.tolist()

    X = np.array([np.array(xi) for xi in tqdm(X)])
    y = codes_df_list
    print("everything has been converted to list")
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)

    print("Logistic Regression about to start")
    clf = OneVsRestClassifier(LogisticRegression(random_state=0)).fit(x_train, y_train)  # OvR splits the data in a way so train with each individual label a document could be, instead of the entire set of labels
    score = clf.score(x_test, y_test)
    print("SCORE HAS BEEN CALCULATED")
    print(score)

    predictions = clf.predict(x_test)
    print(classification_report(y_test, predictions))  # prediction report for Logistic Regression

    # the following code is all for the different machine learning models.
    # the main difference is lack of flatten as the models could take in 3D data.
    X = df.TEXT.to_list()
    print("csv TEXT has been converted into a list")

    invalid = ['TEXT', 'CODES', 'LENGTH']
    codes_df = df[df.columns.difference(invalid)]
    codes_df_list = codes_df.values.tolist()

    X = np.array([np.array(xi) for xi in tqdm(X)])
    y = codes_df_list
    print("everything has been converted to list")
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=0)

    swam_caml(x_train, y_train, x_test, y_test)
    text_cnn(x_train, y_train, x_test, y_test)
    caml(x_train, y_train, x_test, y_test)


main()
