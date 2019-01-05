from multiprocessing import Pool

from sklearn.datasets import fetch_20newsgroups

from pyartm.datasets import sklearn_dataset
from pyartm.optimizations import thetaless
from pyartm.optimizations import default
from pyartm import regularizers

import manager


SAMPLES = 100
ITERS_COUNT = 100


if __name__ == '__main__':
    categories = [
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ]
    dataset_train = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    (
        n_dw_matrix_doc_train, token_2_num_doc_train,
        _, doc_targets_doc_train
    ) = sklearn_dataset.prepare(dataset_train)

    dataset_test = fetch_20newsgroups(
        subset='test',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )
    (
        n_dw_matrix_doc_test, token_2_num_doc_test,
        _, doc_targets_doc_test
    ) = sklearn_dataset.prepare(dataset_test, token_2_num=token_2_num_doc_train)

    args_list = list()
    for T in [10, 25]:
        for phi_alpha in [-0.1, 0., 0.1]:
            for theta_alpha in [-0.1, 0., 0.1]:
                regularization_list = [
                    regularizers.Additive(phi_alpha, theta_alpha)
                ] * ITERS_COUNT
                args_list.append((
                    n_dw_matrix_doc_train, doc_targets_doc_train,
                    n_dw_matrix_doc_test, doc_targets_doc_test,
                    default.Optimizer(regularization_list), T, SAMPLES,
                    '20newsgroups_doc_experiment/20news_{}t_base_{}_{}.pkl'.format(
                        T, phi_alpha, theta_alpha
                    )
                ))
                for use_B_cheat in [False, True]:
                    args_list.append((
                        n_dw_matrix_doc_train, doc_targets_doc_train,
                        n_dw_matrix_doc_test, doc_targets_doc_test,
                        thetaless.Optimizer(regularization_list), T, SAMPLES,
                        '20newsgroups_doc_experiment/20news_{}t_artm_{}_{}_{}.pkl'.format(
                            T, phi_alpha, theta_alpha, use_B_cheat
                        )
                    ))

    Pool(processes=5).map(
        manager.perform_doc_experiment, args_list
    )
