import argparse
import sys
import pandas as pd
import nltk
from nltk.corpus import treebank
from nltk import Production, Nonterminal
from nltk.grammar import induce_pcfg
from nltk.parse import ViterbiParser
from nltk.tokenize import word_tokenize

import numpy as np

from collections import Counter


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str)
    parser.add_argument('--dataset_name', type=str)
    return parser


def add_unknown(rules):
    rules_to_add = set()
    for r in rules:
        # add a rule if the existing rule is lexical
        if r.is_lexical():
            rules_to_add.add(Production(r.lhs(), ['UNK']))
    rules.update(rules_to_add)


def parse_sentence(data):
    # trouver les règles à partir du ptb
    productions = []
    vocabulary = set()
    for item in treebank.fileids():
        for tree in treebank.parsed_sents(item):
            vocabulary.update(tree.leaves())
            productions += tree.productions()

    rules = Counter(productions)

    add_unknown(rules)

    # create grammar
    S = Nonterminal('S')
    grammar = induce_pcfg(S, list(rules.elements()))

    print("Grammar created")

    # create parser
    viterbi = ViterbiParser(grammar)
    prob = np.empty(data.shape[0], dtype=float)
    for i, sentence in enumerate(data['sentence']):
        # tokenize
        words = [
            w if w in vocabulary else 'UNK' for w in word_tokenize(sentence)]

        try:
            parse_sent = viterbi.parse(words)
            for t in parse_sent:
                prob[i] = str(t).split("p=")[-1].rstrip(")")
        except:
            prob[i] = 0

    print("Parses created")

    return prob


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


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])
    train_file = args.train_file
    dev_file = args.dev_file
    dataset_name = args.dataset_name

    # lire les fichiers
    train_data = read_data(train_file, dataset_name)
    test_data = read_data(dev_file, dataset_name)

    # parser les phrases
    prob_parse_train = parse_sentence(train_data)
    prob_parse_train.savetxt(
        r"C:\Users\Naima\OneDrive - Universite de Montreal\TALN\Projet 1\data\prob_parse_train_cola")
    print("Parsed sentences train saved")
    prob_parse_test = parse_sentence(test_data)
    prob_parse_test.savetxt(
        r"C:\Users\Naima\OneDrive - Universite de Montreal\TALN\Projet 1\data\prob_parse_test_cola")
    print("Parsed sentences test saved")
