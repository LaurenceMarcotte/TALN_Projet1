import argparse
import sys
import os
from collections import Counter
import pickle

import numpy as np
import pandas as pd
import torch
import torchtext
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
import gensim.parsing as parsing

from BiLSTM import LSTM


class Text_dataset(Dataset):
    def __init__(self, dataset_name, path, file_name, max_length, preprocess):
        self.path = path
        self.file = file_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        # lecture fichier sous forme tsv
        self.data = pd.read_csv(os.path.join(self.path, self.file), sep='\t')

        # standardization des données, i.e. création des colonnes 'sentence' et 'label' contenant les phrases exemples et
        # les classes associées
        if self.dataset_name == 'cola':
            self.data.columns = ["id", "label", "label2", "sentence"]
        elif self.dataset_name == 'qqp':
            self.data = self.data.rename(columns={'question1': 'sentence', 'question2': 'sentence2', 'is_duplicate': 'label'})

        self.preprocess(preprocess.setdefault('lower', True), preprocess.setdefault('remove_punc', True),
                        preprocess.setdefault('remove_numeric', True))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sentence = self.data['sequence'].iloc[item]
        label = self.data['label'].iloc[item]
        sample = {'seq': sentence, 'label': label}
        if self.dataset_name == 'qqp':
            sample = {'seq': sentence, 'seq2': self.data['sequence2'].iloc[item], 'label': label}
        return sample

    def to_sequence(self, vocab):
        self.data['sequence'] = self.data['tokens'].apply(lambda s: torch.tensor(vocab.lookup_indices(s), dtype=torch.long))
        if self.dataset_name == 'qqp':
            self.data['sequence2'] = self.data['tokens2'].apply(lambda s: torch.tensor(vocab.lookup_indices(s), dtype=torch.long))

    def get_sentences(self):
        if self.dataset_name == 'qqp':
            return list(self.data['tokens'].values) + list(self.data['tokens2'].values)
        return list(self.data['tokens'].values)

    def pad_sentence(self, s):
        if len(s) <= self.max_length:
            for _ in range(self.max_length-len(s)):
                s.append('<pad>')
        else:
            diff = len(s) - self.max_length
            s = s[diff:]
        return s

    def preprocess(self, lower=True, remove_punc=True, remove_numeric=True):
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


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./data/QQP')
    parser.add_argument('--dataset_name', type=str, default='qqp')
    parser.add_argument('--file_type', type=str, default='tsv')
    parser.add_argument('--vocab_saved', action='store_true', default=True)
    parser.add_argument('--vocab_to_save', action='store_true', default=True)
    parser.add_argument('--hidden_size', type=int, default=150)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--progress_bar', action='store_true', default=True)

    return parser


def create_vocabulary_and_embedding_matrix(train, test, embeddings):
    counter = Counter([t for s in train for t in s]+[t for s in test for t in s])
    voc = torchtext.vocab.build_vocab_from_iterator(train+test)
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

    embedding_matrix=torch.from_numpy(embedding_matrix)
    return voc, embedding_matrix


def train(epoch, model, dataloader, optimizer, batch_size, progress_bar, print_every):

    model.train()

    losses = []
    total_iters = 0
    model.float()
    nllloss = torch.nn.NLLLoss()
    for idx, batch in enumerate(
            tqdm(
                dataloader, desc="Epoch {0}".format(epoch), disable=(not progress_bar)
            )
    ):
        # batch = to_device(batch, args.device)
        optimizer.zero_grad()

        if model.similarity_task:
            hidden_states, hidden_states2 = model.initial_states(batch["seq"].shape[0])
            log_probas, _ = model(batch["seq"], hidden_states, batch['seq2'], hidden_states2)
        else:
            hidden_states = model.initial_states(batch["seq"].shape[0])
            log_probas, _ = model(batch["seq"], hidden_states)


        loss = nllloss(log_probas, batch["label"])
        losses.append(loss.item())
        # TODO: Evaluer accuracy
        loss.backward()
        optimizer.step()

        total_iters += 1

        if idx % print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")

    mean_loss = np.mean(losses)
    mean_loss /= batch_size * dataloader.dataset.max_length

    # tqdm.write(f"== [TRAIN] Epoch: {epoch}, Perplexity: {perplexity:.3f} ==>")

    # return mean_loss, perplexity
    return mean_loss


def evaluate(epoch, model, dataloader, batch_size, progress_bar, print_every, mode="val"):
    model.eval()
    losses = []

    total_loss = 0.0
    total_iters = 0

    with torch.no_grad():
        for idx, batch in enumerate(
                tqdm(dataloader, desc="Evaluation", disable=(not progress_bar))
        ):

            if model.similarity_task:
                hidden_states, hidden_states2 = model.initial_states(batch["seq"].shape[0])
                log_probas, _ = model(batch["seq"], hidden_states, batch["seq2"], hidden_states2)
            else:
                hidden_states = model.initial_states(batch["seq"].shape[0])
                log_probas = model(batch["seq"], hidden_states)

            loss = model.loss(log_probas, batch["label"])
            losses.append(loss.item())

            total_loss += loss.item()
            total_iters += batch["sequence"].shape[1]

            if idx % print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}"
                )

        mean_loss = np.mean(losses)
        mean_loss /= batch_size * dataloader.dataset.max_length

    return mean_loss


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    path = args.path
    dataset_name = args.dataset_name
    file_type = args.file_type

    # hyper-parameters:
    lr = 1e-4
    batch_size = args.batch_size
    dropout_keep_prob = 0.5
    max_document_length = 100  # each sentence has until 100 words
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


    print('Loading embeddings')
    embeddings = torchtext.vocab.FastText('simple')
    embedding_size = embeddings.dim

    preprocess = {'lower': True, 'remove_punc': True, 'remove_numeric': True}

    print('Loading data and preprocessing')
    train_data = Text_dataset(dataset_name, path, 'train.'+file_type, max_document_length, preprocess)
    test_data = Text_dataset(dataset_name, path, 'dev.'+file_type, max_document_length, preprocess)

    print('Creating vocab and embedding matrix')
    if saved:
        vocab_file = open(dataset_name+'_vocab.p', 'rb')
        vocab = pickle.load(vocab_file)
        emb_matrix_file = open(dataset_name+'_emb_matrix.p', 'rb')
        embedding_matrix=pickle.load(emb_matrix_file)
    else:
        sentence_train = train_data.get_sentences()
        sentence_test = test_data.get_sentences()
        vocab, embedding_matrix = create_vocabulary_and_embedding_matrix(sentence_train, sentence_test, embeddings)

    print('Vocabulary size:', len(vocab))

    if save:
        vocab_file = open(dataset_name+'_vocab.p', 'wb')
        pickle.dump(vocab, vocab_file)

        emb_matrix_file = open(dataset_name+'_emb_matrix.p', 'wb')
        pickle.dump(embedding_matrix, emb_matrix_file)

    print('Transforming sentences to sequences of ints')
    train_data.to_sequence(vocab)
    test_data.to_sequence(vocab)

    train_size = int(len(train_data)*dev_size)
    dev_size = len(train_data) - train_size
    train_data, valid_data = random_split(train_data, [train_size, dev_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # initialize model
    print('Initializing model')
    similarity_task = False
    if dataset_name == 'qqp':
        similarity_task = True
    model = LSTM(len(vocab), embedding_size, hidden_size, num_layers, pad_index=1, _embedding_weight=embedding_matrix,
                 similarity_task=similarity_task)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        tqdm.write(f"====== Epoch {epoch} ======>")

        loss=train(epoch, model, train_loader, optimizer, batch_size, progress_bar, print_every)
        train_losses.append(loss)

        loss = evaluate(epoch, model, dev_loader, batch_size, progress_bar, print_every)
        valid_losses.append(loss)










