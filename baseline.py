import sys
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--dev_file', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--dataset_name', type=str)
    return parser


def read_data(file_name, dataset_name):
    data = pd.read_csv(file_name, sep='\t')

    if dataset_name == 'cola':
        pass
    elif dataset_name == 'qqp':
        pass
    elif dataset_name == 'sst':
        pass

    return data


def train_model():
    pass


def test_model():
    pass


if __name__ == '__main__':
    parser = make_parser()

    args = parser.parse_args(sys.argv[1:])
