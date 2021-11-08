# %%
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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
    parser.add_argument('--representation', type=str)
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


def train_model_frequency(X_train, y_train):
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(binary=True)),
            ("clf", MLPClassifier()),
        ]
    )

    # hyperparamètre à explorer
    parameters = {'vect__ngram_range': [[(1, 1)], [(1, 2)], [(2, 2)], [(3, 3)]], 'clf__hidden_layer_sizes': [(50,), (100, 50), (100, 50, 50), (100, 50, 50, 50), (100, 50, 50, 50, 50)],
                  'clf__alpha': ([1], [0.1], [0.01], [0.001], [0.0001], [0])}

    params = list(ParameterGrid(parameters))
    #cv = RepeatedStratifiedKFold()

    gs_clf = GridSearchCV(pipeline, param_grid=params,
                          cv=5, n_jobs=-1, refit=True, scoring='accuracy')

    model = gs_clf.fit(X_train, y_train)

    print("Best Score: %s" % model.best_score_)
    print("Best Hyperparameters: %s" % model.best_params_)

    return model


def train_model(model, X_train, y_train, max_nodes=100, nodes_step=5):
    # exploration des hyperparamètres
    # code inspiré de https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
    # define evaluation
    #cv = RepeatedStratifiedKFold()

    # define search space
    space = dict()
    space["hidden_layer_sizes"] = [
        (50,), (100, 50), (100, 50, 50), (100, 50, 50, 50), (100, 50, 50, 50, 50)]
    space["alpha"] = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])

    # define search
    search = RandomizedSearchCV(
        model, space, n_iter=20, scoring="accuracy", n_jobs=-1, cv=5, verbose=1, random_state=1)
    # executer search
    result = search.fit(X_train, y_train)
    # summarize result
    print("Best Score: %s" % result.best_score_)
    print("Best Hyperparameters: %s" % result.best_params_)

    return result


def test_sentence(sentence):
    # split la phrase en tokens
    # mettre à 0 si: termine par conjonction de coordination, deux mots pareils qui se suivent, commence par un pronom objet
    tokenized_sentence = sentence.lower().split(" ")
    object_pronouns = {"me", "him", "us"}
    reflexive_pronouns = {"them", "myself", "yourself", "himself",
                          "herself", "itself", "ourselves", "yourselves", "themselves"}
    possessive_pronouns = {"mine", "yours", "ours", "yours", "theirs"}
    coordinating_conjunctions = {"and", "but", "or"}
    punctuation = {"!", "?", "."}

    sentence_array = np.array(tokenized_sentence, dtype=str)
    if np.any(sentence_array[:-1] == sentence_array[1:]):
        return 0

    elif tokenized_sentence[0] in object_pronouns.union(reflexive_pronouns).union(possessive_pronouns):
        return 0

    elif tokenized_sentence[-1] in punctuation:
        if tokenized_sentence[-2] in coordinating_conjunctions:
            return 0

    elif tokenized_sentence[-1] in coordinating_conjunctions:
        return 0

    return 1


if __name__ == "__main__":
    parser = make_parser()

    args = parser.parse_args(sys.argv[1:])

    train_file = args.train_file
    dev_file = args.dev_file
    dataset_name = args.dataset_name
    representation = args.representation

    print("Reading data, preprocessing...")

    # lire les fichiers
    train_data = read_data(train_file, dataset_name)
    test_data = read_data(dev_file, dataset_name)
    if representation == "semantic":
        # # représentations word2vec
        train_corpus = list(read_corpus(train_data))
        test_corpus = list(read_corpus(test_data))

        X_train = convert_to_matrix(train_corpus, train_data)
        y_train = train_data["label"].to_numpy()
        X_test = convert_to_matrix(test_corpus, test_data)
        y_test = test_data["label"].to_numpy()

        #X_train.append(X_train, prob_parse_train, axis=1)
        #X_test.append(X_test, prob_parse_test, axis=1)

    if representation == "frequency":
        X_train = train_data["sentence"]
        y_train = train_data["label"]
        X_test = test_data["sentence"]
        y_test = test_data["label"]

    print("Training model...")

    # fit le model aux données de train

    if representation == "frequency":
        mlp = train_model_frequency(X_train, y_train)

    if representation == "semantic":
        if dataset_name == "cola":
            mlp = MLPClassifier(
                max_iter=2000, hidden_layer_sizes=(50, 50, 50, 50), alpha=0, activation="logistic", verbose=False)
            mlp.fit(X_train, y_train)
        else:
            mlp = train_model(mlp, X_train, y_train)

    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    # Tests manuels

    # positions des phrases agrammaticales
    idx_where_0train = [i for i in range(
        train_data.shape[0]) if not test_sentence(train_data["sentence"].loc[i])]
    idx_where_0test = [i for i in range(
        test_data.shape[0]) if not test_sentence(test_data["sentence"].loc[i])]
    # On met manuellement ces phrases à 0
    predict_train[idx_where_0train] = 0
    predict_test[idx_where_0test] = 0

    # print les résultats
    print(confusion_matrix(y_train, predict_train))
    print(classification_report(y_train, predict_train))
    print("###################################")
    print(confusion_matrix(y_test, predict_test))
    print(classification_report(y_test, predict_test))


# %%
