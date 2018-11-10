import pickle
import os

from sklearn.datasets import fetch_20newsgroups

from artm import datasets
from artm.datasets import sklearn_dataset
from artm.datasets import nips
from artm.datasets import twitter_sentiment140

ARTM_DIR = os.path.dirname(os.path.realpath(datasets.__file__))
ARTM_RESOURCES = os.path.join(ARTM_DIR, 'resources')
NIPS_PATH = os.path.join(
    os.path.expanduser('~'), 'artm_data', 'NIPS_1987-2015.csv'
)
TWITTER_SENTIMENT140_PATH = os.path.join(
    os.path.expanduser('~'), 'artm_data', 'training.1600000.processed.noemoticon.csv'
)


def set_nips_path(path):
    global NIPS_PATH
    NIPS_PATH = path


def set_twitter_sentiment140_path(path):
    global TWITTER_SENTIMENT140_PATH
    TWITTER_SENTIMENT140_PATH = path


if not os.path.exists(ARTM_RESOURCES):
    os.makedirs(ARTM_RESOURCES)


def get_resource_path(name):
    return os.path.join(ARTM_RESOURCES, name)


def get_20newsgroups(categories, min_occurrences=3, train_test_split=None, subset='all'):
    path = get_resource_path('20newsgroups_subset_{}_{}_{}_{}.pkl'.format(
        subset, min_occurrences, train_test_split,
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
            train_test_split=train_test_split
        )
        with open(path, 'w') as resource_file:
            pickle.dump(data, resource_file)
    return data


def get_nips(test_proportion=None, dataset_path=NIPS_PATH):
    path = get_resource_path('nips_{}_{}.pkl'.format(
        os.path.realpath(dataset_path).replace(os.path.sep, '_'),
        test_proportion
    ))
    if os.path.exists(path):
        with open(path, 'r') as resource_file:
            data = pickle.load(resource_file)
    else:
        data = nips.prepare(dataset_path, test_proportion=test_proportion)
        with open(path, 'w') as resource_file:
            pickle.dump(data, resource_file)
    return data


def get_twitter_sentiment140(
    test_proportion=None, min_docs_occurrences=3,
    dataset_path=TWITTER_SENTIMENT140_PATH
):
    path = get_resource_path('twitter_sentiment140_{}_{}_{}.pkl'.format(
        os.path.realpath(dataset_path).replace(os.path.sep, '_'),
        test_proportion, min_docs_occurrences
    ))
    if os.path.exists(path):
        with open(path, 'r') as resource_file:
            data = pickle.load(resource_file)
    else:
        data = twitter_sentiment140.prepare(
            dataset_path,
            test_proportion=test_proportion,
            min_docs_occurrences=min_docs_occurrences
        )
        with open(path, 'w') as resource_file:
            pickle.dump(data, resource_file)
    return data
