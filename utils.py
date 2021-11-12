from collections import Counter

import numpy as np
import torch
import torchtext


def to_device(tensors, device):
    """
    Envoie les tensors sur le device demandé
    :param tensors: les tenseurs à transférer sur un autre device
    :param device: cpu or cuda
    :return: les tenseurs transférés
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))


def create_vocabulary_and_embedding_matrix(train, test, embeddings):
    """
    Crée le vocabulaire qui lie un mot vers un indice et la matrix des plongements de mot qui lie un indice vers un
    vecteur.
    :param train: le dataset d'entraînement contenant les phrases sous forme de token
    :param test: le dataset de test contenant les phrases sous forme de token
    :param embeddings: les plongements de mots de FastTest
    :return: vocabulaire et la matrice des plongements de mots qui sera à donner au modèle RNN
    """
    counter = Counter([t for s in train for t in s] + [t for s in test for t in s])
    voc = torchtext.vocab.build_vocab_from_iterator(train + test)
    embedding_voc = embeddings.itos

    for word, i in voc.get_stoi().items():
        if word not in embedding_voc:
            del counter[word]

    voc = torchtext.vocab.build_vocab_from_iterator([list(counter)], specials=['<unk>', '<pad>'])
    voc.set_default_index(0)
    embedding_matrix = np.zeros((len(voc), embeddings.dim))
    for word, i in voc.get_stoi().items():
        vector = embeddings[word]
        embedding_matrix[i] = vector

    embedding_matrix = torch.from_numpy(embedding_matrix)
    return voc, embedding_matrix
