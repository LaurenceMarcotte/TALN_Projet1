import argparse
import sys
import os
from collections import Counter
import pickle

from comet_ml import Experiment
import numpy as np
import pandas as pd
import torch
import torchtext
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import gensim.parsing as parsing

from BiLSTM import LSTM


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
        sample = {'seq': sentence, 'label': label}
        if self.dataset_name == 'qqp':
            sample = {'seq': sentence, 'seq2': self.data['sequence2'].iloc[item], 'label': label}
        return sample

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

    def preprocess(self, lower=True, remove_punc=True, remove_numeric=True):
        """
        On preprocess les données
        :param lower: True si on veut mettre les mots en minuscule
        :param remove_punc: True si on veut retirer la ponctuation
        :param remove_numeric: True si on veut retirer les chiffres
        :return:
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

        if self.dataset_name == 'qqp':
            self.data['tokens2'] = self.data['sentence2'].apply(lambda s: parsing.preprocess_string(s, filters))
            self.data['tokens2'] = self.data['tokens2'].apply(lambda s: self.pad_sentence(s))


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


def train(epoch, model, dataloader, optimizer, batch_size, progress_bar, print_every, log_comet=False):
    """
    Fait une époque d'entraînement
    :param epoch: l'époque à laquel on est rendu
    :param model: le modèle qu'on entraîne
    :param dataloader: le dataloader qui contient les batchs de données
    :param optimizer: l'optimizateur (AdamW)
    :param batch_size: la taille des batchs
    :param progress_bar: si on affiche la bar de progrès ou non
    :param print_every: à combien d'itérations est-ce qu'on affiche
    :param log_comet: True si on log les métriques sur comet.ml
    :return: la loss moyenne et la performance moyenne par rapport à l'époque
    """
    model.train()

    accuracy_train = []
    losses = []
    total_iters = 0
    model.float()
    nllloss = torch.nn.NLLLoss()  # la loss qu'on va utiliser
    for idx, batch in enumerate(
            tqdm(
                dataloader, desc="Epoch {0}".format(epoch), disable=(not progress_bar)
            )
    ):
        # batch = to_device(batch, args.device)
        optimizer.zero_grad()

        # si on fait une tâche de similarité entre 2 phrases, on a besoin de 2 phrases comme input
        if model.similarity_task:
            hidden_states, hidden_states2 = model.initial_states(batch["seq"].shape[0])
            log_probas, _ = model(batch["seq"], hidden_states, batch['seq2'], hidden_states2)
        else:
            hidden_states = model.initial_states(batch["seq"].shape[0])
            log_probas, _ = model(batch["seq"], hidden_states)

        loss = nllloss(log_probas, batch["label"])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()  # on optimize ici

        # eval accuracy
        pred = torch.argmax(log_probas, dim=1)
        accuracy = torch.where(pred == batch['label'], 1, 0).sum() / batch_size
        total_iters += 1
        accuracy_train.append(accuracy)

        if idx % print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}, Acc: {accuracy.item():.5f}")

        # ajoute les métriques sur comet.ml
        if log_comet:
            experiment.log_metric("train_accuracy", accuracy.item(), step=idx + len(dataloader) * epoch, epoch=epoch)
            experiment.log_metric("train_loss", loss.item(), step=idx + len(dataloader) * epoch, epoch=epoch)

    mean_loss = np.mean(losses)
    mean_loss /= batch_size
    mean_acc = np.mean(accuracy_train)

    tqdm.write(f"== [TRAIN] Epoch: {epoch}, mean_Loss: {mean_loss:.5f}, Acc:{mean_acc:.5f} ==>")

    # return mean_loss, perplexity
    return mean_loss, mean_acc


def evaluate(epoch, model, dataloader, batch_size, progress_bar, print_every, mode="val", log_comet=False,
             log_confusion_matrix=False):
    """
    Évalue le modèle sur un dataset de validation
    :param epoch: l'époque à laquelle on est rendue
    :param model: le modèle qu'on entraîne
    :param dataloader: le dataloader qui contient les batchs de données
    :param batch_size: la taille de la batch
    :param progress_bar: si on affiche la bar de progrès ou non
    :param print_every: à combien d'itérations on affiche
    :param mode: si on est en mode évaluation ou test
    :param log_comet: True si on log les métriques sur comet.ml
    :param log_confusion_matrix: si on crée une matrice de confusion ou non
    :return: la loss et la performance moyenne sur le dataset
    """
    model.eval()
    losses = []

    total_loss = 0.0
    total_iters = 0
    accuracy_eval = []
    nllloss = torch.nn.NLLLoss()
    confusion_matrix = torch.zeros(2, 2)
    with torch.no_grad():
        for idx, batch in enumerate(
                tqdm(dataloader, desc="Evaluation", disable=(not progress_bar))
        ):

            if model.similarity_task:
                hidden_states, hidden_states2 = model.initial_states(batch["seq"].shape[0])
                log_probas, _ = model(batch["seq"], hidden_states, batch["seq2"], hidden_states2)
            else:
                hidden_states = model.initial_states(batch["seq"].shape[0])
                log_probas, _ = model(batch["seq"], hidden_states)

            loss = nllloss(log_probas, batch["label"])
            losses.append(loss.item())

            total_loss += loss.item()
            total_iters += batch["seq"].shape[1]
            # eval accuracy
            pred = torch.argmax(log_probas, dim=1)
            accuracy = torch.where(pred == batch['label'], 1, 0).sum() / batch_size
            accuracy_eval.append(accuracy)

            if log_confusion_matrix:
                for t, p in zip(batch['label'].view(-1), pred.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

            if idx % print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}, Acc:{accuracy.item():.5f}"
                )

        mean_loss = np.mean(losses)
        mean_loss /= batch_size
        mean_acc = np.mean(accuracy_eval)

        if log_comet:
            experiment.log_metric("val_accuracy", mean_acc, epoch=epoch)
            experiment.log_metric("val_loss", mean_loss, epoch=epoch)

        if log_confusion_matrix:
            experiment.log_confusion_matrix(matrix=confusion_matrix)

        tqdm.write(f"== [VAL] Epoch: {epoch}, mean_Loss: {mean_loss:.5f}, Acc:{mean_acc:.5f} ==>")

    return mean_loss, mean_acc


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/CoLA', help='Folder to where the data is saved')
    parser.add_argument('--dataset_name', type=str, default='cola', help='name of the dataset')
    parser.add_argument('--file_type', type=str, default='tsv', help="Type of file of the dataset (.csv, .tsv)")
    parser.add_argument('--max_sentence_length', type=int, default=100, help="The maximum length for a sentence")
    parser.add_argument('--vocab_saved', action='store_true', default=False,
                        help="Use if vocab has been computed before"
                             " and saved")
    parser.add_argument('--vocab_to_save', action='store_true', default=False, help="Use if you want to save the "
                                                                                   "vocabulary and embedding matrix")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--hidden_size', type=int, default=150, help="The size of the hidden_size for the lstm")
    parser.add_argument('--num_layers', type=int, default=1, help="The number of layers for the lstm")
    parser.add_argument('--batch_size', type=int, default=64, help="The size of the batch for training and eval")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="The value for the weight decay in the optimizer AdamW")
    parser.add_argument('--epochs', type=int, default=3, help="The number of epochs to train")
    parser.add_argument('--print_every', type=int, default=10, help="After how many steps do you want to print info")
    parser.add_argument('--progress_bar', action='store_true', default=False, help="If you want to show the progress")
    parser.add_argument('--comet', action='store_true', default=False,
                        help="Use if you want to log metrics on comet.ml")

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    comet = args.comet

    # si on veut log l'expérience sur comet.ml
    if comet:
        experiment = Experiment(
            api_key="hdTbgQGSQDtOoxv7ZbHrOx2OU",
            project_name="projet-1-nlp",
            workspace="floui",
        )

    path = args.path
    dataset_name = args.dataset_name
    file_type = args.file_type

    # hyper-parameters:
    lr = args.learning_rate
    batch_size = args.batch_size
    dropout_keep_prob = 0.5
    max_document_length = args.max_sentence_length  # each sentence has until 100 words
    dev_size = 0.8  # split percentage to train\validation data
    # max_size = 2e5  # maximum vocabulary size
    seed = 42
    num_classes = 2
    save = args.vocab_to_save
    saved = args.vocab_saved
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    weight_decay = args.weight_decay
    epochs = args.epochs
    print_every = args.print_every
    progress_bar = args.progress_bar

    # log les hyperparamètres sur comet.ml
    if comet:
        hyper_params = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout": dropout_keep_prob,
            "max_doc_length": max_document_length,
            "seed": seed,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "weight_decay": weight_decay,
            "dataset_name": dataset_name,
        }
        experiment.log_parameters(hyper_params)

    print('Loading embeddings')
    embeddings = torchtext.vocab.FastText('simple')
    embedding_size = embeddings.dim

    preprocess = {'lower': True, 'remove_punc': True, 'remove_numeric': True}

    print('Loading data and preprocessing')
    train_data = TextDataset(dataset_name, path, 'train.' + file_type, max_document_length, preprocess)
    test_data = TextDataset(dataset_name, path, 'dev.' + file_type, max_document_length, preprocess)

    print('Creating vocab and embedding matrix')
    # si on a déjà créé et enregistré le vocabulaire et la matrice d'embedding, on va simplement les chercher plutôt
    # que de les créer à nouveau
    if saved:
        vocab_file = open(dataset_name + '_vocab.p', 'rb')
        vocab = pickle.load(vocab_file)
        emb_matrix_file = open(dataset_name + '_emb_matrix.p', 'rb')
        embedding_matrix = pickle.load(emb_matrix_file)
    else:
        sentence_train = train_data.get_sentences()
        sentence_test = test_data.get_sentences()
        vocab, embedding_matrix = create_vocabulary_and_embedding_matrix(sentence_train, sentence_test, embeddings)

    print('Vocabulary size:', len(vocab))

    # on sauvegarde le vocabulaire et la matrice d'embedding
    if save:
        vocab_file = open(dataset_name + '_vocab.p', 'wb')
        pickle.dump(vocab, vocab_file)

        emb_matrix_file = open(dataset_name + '_emb_matrix.p', 'wb')
        pickle.dump(embedding_matrix, emb_matrix_file)

    print('Transforming sentences to sequences of ints')
    train_data.to_sequence(vocab)
    test_data.to_sequence(vocab)

    # on génère l'ensemble de validation pour l'entraînement
    train_size = int(len(train_data) * dev_size)
    dev_size = len(train_data) - train_size
    train_data, valid_data = random_split(train_data, [train_size, dev_size])

    # on génère les dataloader de chaque dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # initialize model
    print('Initializing model')
    similarity_task = False
    num_add_feature = 0
    if dataset_name == 'qqp':
        similarity_task = True
        num_add_feature = 1
    model = LSTM(len(vocab), embedding_size, hidden_size, num_layers, pad_index=1, _embedding_weight=embedding_matrix,
                 dropout_prob=dropout_keep_prob, num_add_feature=num_add_feature, similarity_task=similarity_task)

    # initialisation de l'optimiseur
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # BOUCLE D'ENTRAÎNEMENT
    # on entraîne sur le nombre d'époque demandé
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    log_confusion_matrix = False
    for epoch in range(epochs):
        tqdm.write(f"====== Epoch {epoch} ======>")

        loss, acc = train(epoch, model, train_loader, optimizer, batch_size, progress_bar, print_every, log_comet=comet)
        train_losses.append(loss)
        train_acc.append(acc)

        if epoch == epochs - 1 and comet:
            log_confusion_matrix = True

        loss, acc = evaluate(epoch, model, dev_loader, batch_size, progress_bar, print_every, log_comet=comet,
                             log_confusion_matrix=log_confusion_matrix)
        valid_losses.append(loss)
        valid_acc.append(acc)

    experiment.end()
