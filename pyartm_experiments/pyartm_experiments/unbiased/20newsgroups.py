# coding: utf-8
from multiprocessing import Pool

from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import default

import manager


if __name__ == '__main__':
    train_n_dw_matrix, test_n_dw_matrix = main_cases.get_20newsgroups([
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ], train_proportion=0.8)[:2]

    args_list = list()
    for T in [10, 30]:
        for tau in [1e7, 1e8, 1.5e8, 2e8, 2.5e8, 3e8, 3.5e8, 4e8, 4.5e8, 5e8]:
            for use_old_phi in [True]:  # [False, True]
                regularization_list = [
                    regularizers.Combination(
                        regularizers.Decorrelator(tau, use_old_phi),
                        regularizers.Additive(-0.01, -0.01),
                    )
                ] * 500
                args_list.append((
                    train_n_dw_matrix, test_n_dw_matrix,
                    default.Optimizer(regularization_list), T, 10,
                    '20news_experiment/20news_{}t_{}_{}.pkl'.format(
                        T, int(tau), use_old_phi
                    )
                ))

    Pool(processes=8).map(manager.perform_experiment, args_list)
