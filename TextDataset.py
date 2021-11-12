import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import gensim.parsing as parsing


class TextDataset(Dataset):
    def __init__(self, dataset_name, path, file_name, max_length, preprocess):
        self.path = path
        self.file = file_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        # lecture fichier sous forme tsv
        self.data = pd.read_csv(os.path.join(self.path, self.file), sep='\t')

        # standardization des données, i.e. création des colonnes 'sentence' et 'label' contenant les phrases
        # exemples et les classes associées
        if self.dataset_name == 'cola':
            self.data.columns = ["id", "label", "label2", "sentence"]
        elif self.dataset_name == 'qqp':
            self.data = self.data.rename(columns={'question1': 'sentence', 'question2': 'sentence2',
                                                  'is_duplicate': 'label'})

        self.preprocess(preprocess.setdefault('lower', True), preprocess.setdefault('remove_punc', True),
                        preprocess.setdefault('remove_numeric', True))

    def __len__(self):
        """
        la longueur du dataset (le nombre d'exemples)
        :return:
        """
        return len(self.data)

    def __getitem__(self, item):
        """
        Va chercher un exemple du dataset correspondant à l'indice item
        :param item: l'indice de l'exemple à aller chercher
        :return: l'exemple demandé
        """
        sentence = self.data['sequence'].iloc[item]
        label = self.data['label'].iloc[item]
        sample = {'seq': sentence, 'label': label, 'id': torch.tensor([item])}
        if self.dataset_name == "cola":
            sample["rule"] = self.data["rule"].iloc[item]
        if self.dataset_name == 'qqp':
            common = self.data['common'].iloc[item]
            sample = {'seq': sentence, 'seq2': self.data['sequence2'].iloc[item], 'label': label,
                      'id': torch.tensor([item]), 'common': common}
        return sample

    def get_common_words(self, row):
        """
        Calcule le nombre de mots en commun entre les deux séquences d'un exemple si self.similarity_task est True
        :param row: la rangée contenant les phrases dont on veut compter le nombre de mots en commun
        :return: un tensor contenant le nombre de mot en commun
        """
        seq1 = set(map(int, row['sequence']))
        seq2 = set(map(int, row['sequence2']))
        if 1 in seq1:
            seq1.remove(1)
        if 1 in seq2:
            seq2.remove(1)
        return torch.tensor([len(seq1.intersection(seq2))])

    def to_sequence(self, vocab):
        """
        Transforme toutes les phrases en séquence de chiffre correspondant à la rangée de la matrice d'embedding
        :param vocab: le vocabulaire du dataset
        """
        self.data['sequence'] = self.data['tokens'].apply(lambda s: torch.tensor(vocab.lookup_indices(s),
                                                                                 dtype=torch.long))
        if self.dataset_name == 'qqp':
            self.data['sequence2'] = self.data['tokens2'].apply(lambda s: torch.tensor(vocab.lookup_indices(s),
                                                                                       dtype=torch.long))
            self.data['common'] = self.data.apply(lambda row: self.get_common_words(row), axis=1)

    def get_sentences(self):
        """
        Pour avoir toutes les phrases du dataset
        :return: une liste de toutes les phrases du dataset
        """
        if self.dataset_name == 'qqp':
            return list(self.data['tokens'].values) + list(self.data['tokens2'].values)
        return list(self.data['tokens'].values)

    def pad_sentence(self, s):
        """
        Pad les phrases jusqu'à la taille maximale avec le token <pad> ou sinon coupe les phrases trop longues
        :param s: la phrase à ajouter les tokens <pad>
        :return: la phrase après l'ajout des <pad>
        """
        if len(s) <= self.max_length:
            for _ in range(self.max_length - len(s)):
                s.append('<pad>')
        else:
            diff = len(s) - self.max_length
            s = s[diff:]
        return s

    def test_sentence(self, sentence):
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
            return torch.tensor([0])

        elif tokenized_sentence[0] in object_pronouns.union(reflexive_pronouns).union(possessive_pronouns):
            return torch.tensor([0])

        elif tokenized_sentence[-1] in punctuation:
            if tokenized_sentence[-2] in coordinating_conjunctions:
                return torch.tensor([0])

        elif tokenized_sentence[-1] in coordinating_conjunctions:
            return torch.tensor([0])

        return torch.tensor([1])

    def preprocess(self, lower=True, remove_punc=True, remove_numeric=True):
        """
        On preprocess les données
        :param lower: True si on veut mettre les mots en minuscule
        :param remove_punc: True si on veut retirer la ponctuation
        :param remove_numeric: True si on veut retirer les chiffres
        """
        filters = []
        if lower:
            filters.append(lambda x: x.lower())

        if remove_punc:
            filters.append(parsing.strip_punctuation)

        filters.append(parsing.strip_multiple_whitespaces)

        if remove_numeric:
            filters.append(parsing.strip_numeric)

        self.data['tokens'] = self.data['sentence'].apply(lambda s: parsing.preprocess_string(s, filters))
        self.data['tokens'] = self.data['tokens'].apply(lambda s: self.pad_sentence(s))

        if self.dataset_name == "cola":
            self.data["rule"] = self.data["sentence"].apply(lambda s: self.test_sentence(s))

        if self.dataset_name == 'qqp':
            self.data['tokens2'] = self.data['sentence2'].apply(lambda s: parsing.preprocess_string(s, filters))
            self.data['tokens2'] = self.data['tokens2'].apply(lambda s: self.pad_sentence(s))
