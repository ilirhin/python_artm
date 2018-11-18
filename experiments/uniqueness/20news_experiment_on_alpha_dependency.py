from multiprocessing import Pool

from artm.datasets import main_cases
from artm import regularizers
from artm.optimizations import default

import manager


if __name__ == '__main__':
    train_n_dw_matrix = main_cases.get_20newsgroups([
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ])[0]
    args_list = list()
    T = 10
    for phi_alpha in [-10 ** (-i) for i in range(30)]:
        for theta_alpha in [-0.1, 0., 0.1]:
            regularization_list = [
                regularizers.Additive(phi_alpha, theta_alpha)] * 100
            args_list.append((
                default.Optimizer(regularization_list), T, 200,
                'alpha_exp/20news_{}t_{}_{}.pkl'.format(
                    T, phi_alpha, theta_alpha
                )
            ))

    Pool(processes=5).map(
        manager.perform_alpha_dependency_experiment, args_list
    )
