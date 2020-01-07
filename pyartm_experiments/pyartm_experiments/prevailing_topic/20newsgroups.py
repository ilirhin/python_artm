import numpy as np

from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import default, thetaless
from pyartm.common import experiments
from pyartm.calculations import metrics

from pyartm_experiments.online import online_default, online_thetaless


def print_matrix(arr):
    for row in arr:
        line = list(map(str, row))
        print(' '.join(line))


if __name__ == '__main__':
    _n_dw_matrix, _, num_2_token, doc_targets = main_cases.get_20newsgroups([
        'comp.sys.mac.hardware',
        'talk.politics.guns',
    ])
    topic_0_indices, topic_1_indices = [],  []
    for index, target in enumerate(doc_targets):
        if target == 0:
            topic_0_indices.append(index)
        elif target == 1:
            topic_1_indices.append(index)

    for balance in [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]:
        n_dw_matrix = _n_dw_matrix[topic_0_indices + topic_1_indices * balance, :]
        regularization_list = [regularizers.Additive(-0.1, 0.)] * 100
        lda_phi, lda_theta = experiments.default_sample(
            n_dw_matrix,
            T=2,
            seed=42,
            optimizer=default.Optimizer(regularization_list, verbose=False)
        )
        thetaless_phi, thetaless_theta = experiments.default_sample(
            n_dw_matrix,
            T=2,
            seed=42,
            optimizer=thetaless.Optimizer(regularization_list, verbose=False)
        )
        online_lda_phi, online_lda_theta = experiments.default_sample(
            n_dw_matrix,
            T=2,
            seed=42,
            optimizer=online_default.Optimizer(
                regularization_list,
                sampling=2000,
                verbose=False
            ),
        )
        online_thetaless_phi, online_thetaless_theta = experiments.default_sample(
            n_dw_matrix,
            T=2,
            seed=42,
            optimizer=online_thetaless.Optimizer(
                regularization_list,
                sampling=2000,
                verbose=False
            )
        )
        for topic_set in metrics.get_top_words(lda_phi, 10):
            print('\n\t'.join(map(num_2_token.get, topic_set)))
            print()
        print('lda')
        print(np.sum(lda_theta[:, 1]) / np.sum(lda_theta[:, 0]))
        print('thetaless')
        print(np.sum(thetaless_theta[:, 1]) / np.sum(thetaless_theta[:, 0]))
        print('online_lda')
        print(np.sum(online_lda_theta[:, 1]) / np.sum(online_lda_theta[:, 0]))
        print('online_thetaless')
        print(np.sum(online_thetaless_theta[:, 1]) / np.sum(online_thetaless_theta[:, 0]))
        print()
        print()
