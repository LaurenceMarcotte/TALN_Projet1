import argparse
import sys
import pickle
import warnings

import numpy as np
import torch
import torchtext
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from BiLSTM import LSTM
from TextDataset import TextDataset
from utils import to_device, create_vocabulary_and_embedding_matrix


def train(epoch, model, dataloader, optimizer, batch_size, progress_bar, print_every, device):
    """
    Fait une époque d'entraînement
    :param device: the device on which to train the model (cpu or cuda)
    :param epoch: l'époque à laquel on est rendu
    :param model: le modèle qu'on entraîne
    :param dataloader: le dataloader qui contient les batchs de données
    :param optimizer: l'optimizateur (AdamW)
    :param batch_size: la taille des batchs
    :param progress_bar: si on affiche la bar de progrès ou non
    :param print_every: à combien d'itérations est-ce qu'on affiche
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
        batch = to_device(batch, device)
        optimizer.zero_grad()

        # si on fait une tâche de similarité entre 2 phrases, on a besoin de 2 phrases comme input
        if model.similarity_task:
            hidden_states, hidden_states2 = model.initial_states(batch["seq"].shape[0])
            log_probas, _ = model(batch["seq"], hidden_states, batch['seq2'], hidden_states2, batch['common'])
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
        accuracy_train.append(accuracy.item())

        if idx % print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}, Acc: {accuracy.item():.5f}")

    mean_loss = np.mean(losses)
    mean_loss /= batch_size
    mean_acc = np.mean(accuracy_train)

    tqdm.write(f"== [TRAIN] Epoch: {epoch}, mean_Loss: {mean_loss:.5f}, Acc:{mean_acc:.5f} ==>")

    return mean_loss, mean_acc


def evaluate(epoch, model, dataloader, batch_size, progress_bar, print_every, best_accuracy, device, mode="val"):
    """
    Évalue le modèle sur un dataset de validation
    :param device: the device on which to evaluate the dataset (cpu or cuda)
    :param best_accuracy: the best_accuracy obtained untill now during evaluation
    :param epoch: l'époque à laquelle on est rendue
    :param model: le modèle qu'on entraîne
    :param dataloader: le dataloader qui contient les batchs de données
    :param batch_size: la taille de la batch
    :param progress_bar: si on affiche la bar de progrès ou non
    :param print_every: à combien d'itérations on affiche
    :param mode: si on est en mode évaluation ou test
    :return: la loss et la performance moyenne sur le dataset
    """
    model.eval()
    losses = []

    total_loss = 0.0
    total_iters = 0
    accuracy_eval = []
    nllloss = torch.nn.NLLLoss()
    confusion_matrix = torch.zeros(2, 2)
    wrong = []
    with torch.no_grad():
        for idx, batch in enumerate(
                tqdm(dataloader, desc="Evaluation", disable=(not progress_bar))
        ):
            batch = to_device(batch, device)
            if model.similarity_task:
                hidden_states, hidden_states2 = model.initial_states(batch["seq"].shape[0])
                log_probas, _ = model(batch["seq"], hidden_states, batch["seq2"], hidden_states2, batch['common'])
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
            accuracy_eval.append(accuracy.item())

            for t, p in zip(batch['label'].view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            if mode == 'test':
                return_sent = torch.where(pred == batch['label'], False, True)
                wrong.append((batch['id'][return_sent], batch['label'][return_sent], log_probas[return_sent]))

            if idx % print_every == 0:
                tqdm.write(
                    f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}, Acc:{accuracy.item():.5f}"
                )

        mean_loss = np.mean(losses)
        mean_loss /= batch_size
        mean_acc = np.mean(accuracy_eval)

        if mean_acc > best_accuracy:
            torch.save(model.state_dict(), './bilstm_best_param.pt')
            best_accuracy = mean_acc

        tqdm.write(f"== [VAL] Epoch: {epoch}, mean_Loss: {mean_loss:.5f}, Acc:{mean_acc:.5f} ==>")

    return mean_loss, mean_acc, best_accuracy, confusion_matrix, wrong


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
    parser.add_argument('--dropout', type=float, default=0.5, help="The probability to keep a neuron active (dropout)")
    parser.add_argument('--epochs', type=int, default=3, help="The number of epochs to train")
    parser.add_argument('--print_every', type=int, default=10, help="After how many steps do you want to print info")
    parser.add_argument('--progress_bar', action='store_true', default=False, help="If you want to show the progress")
    parser.add_argument('--device', choices=['cpu', 'cuda'],
                        help="The device on which to train de model, either cpu or "
                             "cuda")

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args(sys.argv[1:])

    path = args.path
    dataset_name = args.dataset_name
    file_type = args.file_type

    # hyper-parameters:
    lr = args.learning_rate
    batch_size = args.batch_size
    dropout_keep_prob = args.dropout
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

    device = args.device

    # Check for the device
    if (device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU (e.g. in the Notebook Settings in Google Colab). "
            'Forcing device="cpu".'
        )
        device = "cpu"

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
    model.to(device)

    # initialisation de l'optimiseur
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # BOUCLE D'ENTRAÎNEMENT
    # on entraîne sur le nombre d'époque demandé
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    log_confusion_matrix = False
    best_accuracy = 0
    for epoch in range(epochs):
        tqdm.write(f"====== Epoch {epoch} ======>")

        loss, acc = train(epoch, model, train_loader, optimizer, batch_size, progress_bar, print_every, device)
        train_losses.append(loss)
        train_acc.append(acc)

        loss, acc, best_accuracy, confusion_matrix, wrong = evaluate(epoch, model, dev_loader, batch_size, progress_bar,
                                                                     print_every, best_accuracy, device)
        valid_losses.append(loss)
        valid_acc.append(acc)
