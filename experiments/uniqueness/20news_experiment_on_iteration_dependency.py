# coding: utf-8
from multiprocessing import Pool

from pyartm.datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import default

import manager


if __name__ == '__main__':
    train_n_dw_matrix, test_n_dw_matrix = main_cases.get_20newsgroups([
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ], train_proportion=0.8)[:2]
    args_list = list()
    phi_alpha = -0.1
    for T in range(3, 16):
        for theta_alpha in [-0.1, 0., 0.1]:
            regularization_list = [
                regularizers.Additive(phi_alpha, theta_alpha)] * 100
            args_list.append((
                train_n_dw_matrix, test_n_dw_matrix,
                default.Optimizer(regularization_list), T, 10,
                'iter_exp/20news_{}t_{}_{}.pkl'.format(
                    T, phi_alpha, theta_alpha
                )
            ))

    Pool(processes=5).map(
        manager.perform_iteration_dependency_experiment, args_list
    )
