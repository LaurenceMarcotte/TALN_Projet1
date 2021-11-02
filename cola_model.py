import scipy
import sys
import argparse
from numpy.lib.function_base import vectorize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor

from gensim.test.utils import datapath, temporary_file
from gensim import utils
import gensim.models
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import classification_report, confusion_matrix

# code inspiré de https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn

# %%
# rendre ça clean éventuellement avec un argparse


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str)
    parser.add_argument('--dataset_name', type=str)
    return parser


train_file = r"C:\Users\Naima\OneDrive - Universite de Montreal\TALN\Projet 1\data\cola_public_1.1\cola_public\tokenized\in_domain_train.tsv"
dev_file = r"C:\Users\Naima\OneDrive - Universite de Montreal\TALN\Projet 1\data\cola_public_1.1\cola_public\tokenized\in_domain_dev.tsv"
dataset_name = "cola"


def read_data(file_name: str, dataset_name: str) -> pd.DataFrame:
    """
    Lecture des fichiers de données
    :param file_name: str, nom du fichier à lire
    :param dataset_name: str, nom du dataset auquel appartient le fichier
    :return: un dataframe pandas contenant une colonne "sentence" qui sont les exemples et une colonne "label" qui sont
    les labels associés aux exemples
    """
    # lecture fichier sous forme tsv
    if dataset_name == "cola":
        data = pd.read_csv(file_name, sep="\t", header=None)
    else:
        data = pd.read_csv(file_name, sep='\t')

    # standardization des données, i.e. création des colonnes 'sentence' et 'label' contenant les phrases exemples et
    # les classes associées
    if dataset_name == 'cola':
        data.columns = ["id", "label", "label2", "sentence"]
    elif dataset_name == 'qqp':
        data['sentence'] = data.apply(
            lambda row: row['question1'] + ' ' + row['question2'], axis=1)
        data = data.rename(columns={'is_duplicate': 'label'})
    elif dataset_name == 'sst':
        # TODO: Renommer les colonnes du dataframe contenant les phrases et les classes en "sentence" et "label" pour
        #  sst
        pass

    return data


# donner en input des représentations word2vec
train_data = read_data(train_file, dataset_name)
test_data = read_data(dev_file, dataset_name)


def read_corpus(data):
    for i, line in enumerate(data["sentence"]):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


train_corpus = list(read_corpus(train_data))
test_corpus = list(read_corpus(test_data))


def convert_to_matrix(corpus, data):
    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=10, min_count=2, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count,
                epochs=model.epochs)
    sentence2vec = [model.infer_vector((data["sentence"][i].split(
        " "))) for i in range(0, len(data["sentence"]))]
    stv = np.array(sentence2vec)

    return stv


# X = sentence
# y = label
X_train = convert_to_matrix(train_corpus, train_data)
y_train = train_data["label"].to_numpy()
X_test = convert_to_matrix(test_corpus, test_data)
y_test = test_data["label"].to_numpy()

# combien de couches cachées?
# fit le model aux données de train
mlp = MLPClassifier(max_iter=1000)
mlp.fit(X_train, y_train.data)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))
print("###################################")
print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))

# %%
