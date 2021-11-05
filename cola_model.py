# %%
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV

import gensim

from sklearn.metrics import classification_report, confusion_matrix

from collections import Counter
from itertools import product

# code inspiré de https://www.pluralsight.com/guides/machine-learning-neural-networks-scikit-learn

# %%


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str)
    parser.add_argument('--dataset_name', type=str)
    return parser


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

    return data


def read_corpus(data):
    for i, line in enumerate(data["sentence"]):
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def convert_to_matrix(corpus, data, vector_size=30):
    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=vector_size, min_count=2, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count,
                epochs=model.epochs)
    sentence2vec = [model.infer_vector((data["sentence"][i].split(
        " "))) for i in range(0, len(data["sentence"]))]
    stv = np.array(sentence2vec)

    return stv


def train_model(model, X_train, y_train, max_nodes=100, nodes_step=5):
    # exploration des hyperparamètres
    # code inspiré de https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
    # define evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search space
    space = dict()
    space["hidden_layer_sizes"] = [(100,), (100,50,), (100,50,100), (50,100,), (25, 50, 100)]
    space["activation"] = ["identity", "logistic", "tanh", "relu"]
    space["solver"] = ["lbfgs", "sgd", "adam"]
    space["alpha"] = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
    # define search
    search = RandomizedSearchCV(
        model, space, n_iter=500, scoring="accuracy", n_jobs=-1, cv=cv, verbose=10, random_state=1)
    # executer search
    result = search.fit(X_train, y_train.data)
    # summarize result
    print("Best Score: %s" % result.best_score_)
    print("Best Hyperparameters: %s" % result.best_params_)

    return result


if __name__ == "__main__":
    parser = make_parser()

    args = parser.parse_args(sys.argv[1:])

    train_file = args.train_file
    dev_file = args.dev_file
    dataset_name = args.dataset_name

    print("Reading data, preprocessing...")

    # lire les fichiers
    train_data = read_data(train_file, dataset_name)
    test_data = read_data(dev_file, dataset_name)

    # représentations word2vec
    train_corpus = list(read_corpus(train_data))
    test_corpus = list(read_corpus(test_data))

    # X = sentence
    # y = label
    X_train = convert_to_matrix(train_corpus, train_data)
    #X_train.append(X_train, prob_parse_train, axis=1)
    y_train = train_data["label"].to_numpy()
    X_test = convert_to_matrix(test_corpus, test_data)
    #X_test.append(X_test, prob_parse_test, axis=1)
    y_test = test_data["label"].to_numpy()

    print("Training model...")

    # fit le model aux données de train
    mlp = MLPClassifier(
        max_iter=2000, hidden_layer_sizes=(100, 5), verbose=True)

    mlp = train_model(mlp, X_train, y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    # print les résultats
    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))
    print("###################################")
    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))


# %%
