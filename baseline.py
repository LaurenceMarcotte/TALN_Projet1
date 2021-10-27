import sys
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
    data = pd.read_csv(file_name, sep='\t')

    # standardization des données, i.e. création des colonnes 'sentence' et 'label' contenant les phrases exemples et
    # les classes associées
    if dataset_name == 'cola':
        # TODO: Renommer les colonnes du dataframe contenant les phrases et les classes en "sentence" et "label" pour cola
        pass
    elif dataset_name == 'qqp':
        data['sentence'] = data.apply(lambda row: row['question1'] + ' ' + row['question2'], axis=1)
        data = data.rename(columns={'is_duplicate': 'label'})
    elif dataset_name == 'sst':
        # TODO: Renommer les colonnes du dataframe contenant les phrases et les classes en "sentence" et "label" pour sst
        pass

    return data


def train_model(X_train, y_train):
    print("Pipeline created")
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", SGDClassifier(loss='log')),
        ]
    )
    # hyperparamètre à explorer
    parameters = {'vect__ngram_range': [[(1, 1)], [(1, 2)], [(2, 2)], [(3, 3)]], 'tfidf__norm': (['l1'], ['l2']),
                  'tfidf__use_idf': [[True]], 'clf__penalty': (['l1'], ['l2']),
                  'clf__alpha': ([1e-5], [1e-4], [1e-6])}
    # création d'une liste contenant un "produit cartésien" des hyperparamètres
    params = list(ParameterGrid(parameters))

    print("Training...")

    # exploration des hyperparamètres
    gs_clf = GridSearchCV(pipeline, param_grid=params, cv=5, n_jobs=-1, refit=True, scoring='accuracy')
    model = gs_clf.fit(X_train, y_train)
    print("Training done")

    return model


def test_model(model, X_test, y_test):
    # Évaluation du modèle sur les données de test

    print("Testing model")
    y_pred = model.predict(X_test)

    acc = classification_report(y_test, y_pred)
    print("Accuracy on dev set", acc)


if __name__ == '__main__':
    parser = make_parser()

    args = parser.parse_args(sys.argv[1:])

    train_file = args.train_file
    dev_file = args.dev_file
    dataset_name = args.dataset_name

    train_data = read_data(train_file, dataset_name)
    test_data = read_data(dev_file, dataset_name)

    model = train_model(train_data['sentence'], train_data['label'])
    test_model(model, test_data['sentence'], test_data['label'])
