import pickle
import os

from sklearn.datasets import fetch_20newsgroups
import numpy as np
import scipy.sparse

import pyartm_datasets
from pyartm_datasets import sklearn_dataset
from pyartm_datasets import nips
from pyartm_datasets import twitter_sentiment140

ARTM_DIR = os.path.dirname(os.path.realpath(pyartm_datasets.__file__))
ARTM_RESOURCES = os.path.join(ARTM_DIR, 'resources')
DATASETS_PATH = os.environ.get(
    'PYARTM_DATASETS_PATH',
    os.path.join(os.path.expanduser('~'), 'pyartm-datasets')
)
NIPS_PATH = os.path.join(DATASETS_PATH, 'NIPS.csv')
TWITTER_SENTIMENT140_PATH = os.path.join(
    DATASETS_PATH, 'twitter-sentiment140.csv'
)
WNTM_MATRIX_DIR_PATH = os.path.join(DATASETS_PATH, 'wntm_matrix')


def set_nips_path(path):
    global NIPS_PATH
    NIPS_PATH = path


def set_twitter_sentiment140_path(path):
    global TWITTER_SENTIMENT140_PATH
    TWITTER_SENTIMENT140_PATH = path


def set_wntm_matrix_path(path):
    global WNTM_MATRIX_DIR_PATH
    WNTM_MATRIX_DIR_PATH = path


if not os.path.exists(ARTM_RESOURCES):
    os.makedirs(ARTM_RESOURCES)


def get_resource_path(name):
    return os.path.join(ARTM_RESOURCES, name)


def get_20newsgroups(categories, min_occurrences=3, train_proportion=None, subset='all'):
    path = get_resource_path('20newsgroups_subset_{}_{}_{}_{}.pkl'.format(
        subset, min_occurrences, train_proportion,
        '_'.join(sorted(categories)))
    )
    if os.path.exists(path):
        with open(path, 'r') as resource_file:
            data = pickle.load(resource_file)
    else:
        data = sklearn_dataset.prepare(
            fetch_20newsgroups(
                subset=subset,
                categories=categories,
                remove=('headers', 'footers', 'quotes')
            ),
            min_occurrences=min_occurrences,
            train_proportion=train_proportion
        )
        with open(path, 'w') as resource_file:
            pickle.dump(data, resource_file)
    return data


def get_nips(train_proportion=None, dataset_path=NIPS_PATH):
    path = get_resource_path('nips_{}_{}.pkl'.format(
        os.path.realpath(dataset_path).replace(os.path.sep, '_'),
        train_proportion
    ))
    if os.path.exists(path):
        with open(path, 'r') as resource_file:
            data = pickle.load(resource_file)
    else:
        data = nips.prepare(dataset_path, train_proportion=train_proportion)
        with open(path, 'w') as resource_file:
            pickle.dump(data, resource_file)
    return data


def get_twitter_sentiment140(
    train_proportion=None, min_docs_occurrences=3,
    dataset_path=TWITTER_SENTIMENT140_PATH
):
    path = get_resource_path('twitter_sentiment140_{}_{}_{}.pkl'.format(
        os.path.realpath(dataset_path).replace(os.path.sep, '_'),
        train_proportion, min_docs_occurrences
    ))
    if os.path.exists(path):
        with open(path, 'r') as resource_file:
            data = pickle.load(resource_file)
    else:
        data = twitter_sentiment140.prepare(
            dataset_path,
            train_proportion=train_proportion,
            min_docs_occurrences=min_docs_occurrences
        )
        with open(path, 'w') as resource_file:
            pickle.dump(data, resource_file)
    return data


def get_wntm_matrix(wntm_matrix_dir=WNTM_MATRIX_DIR_PATH):
    path = get_resource_path('wntm_matrix_{}.pkl'.format(
        os.path.realpath(wntm_matrix_dir).replace(os.path.sep, '_'),
    ))
    if os.path.exists(path):
        with open(path, 'r') as resource_file:
            n_dw_matrix = pickle.load(resource_file)
    else:
        data = np.load(os.path.join(wntm_matrix_dir, 'data.npy'))
        indices = np.load(os.path.join(wntm_matrix_dir, 'indices.npy'))
        indptr = np.load(os.path.join(wntm_matrix_dir, 'indptr.npy'))

        n_dw_matrix = scipy.sparse.csr_matrix((data, indices, indptr))
        n_dw_matrix.eliminate_zeros()
        with open(path, 'w') as resource_file:
            pickle.dump(n_dw_matrix, resource_file)
    return n_dw_matrix
