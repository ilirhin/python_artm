# coding: utf-8
from multiprocessing import Pool

from artm.datasets import main_cases
from artm import regularizers
from artm.optimizations import default

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
        for tau in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
            for use_old_phi in [False, True]:
                regularization_list = [
                    regularizers.Decorrelator(tau, use_old_phi)] * 100
                args_list.append((
                    train_n_dw_matrix, test_n_dw_matrix,
                    default.Optimizer(regularization_list), T, 100,
                    '20news_experiment/20news_{}t_{}_{}.pkl'.format(
                        T, int(tau), use_old_phi
                    )
                ))

    Pool(processes=8).map(manager.perform_experiment, args_list)
