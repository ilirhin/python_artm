from multiprocessing import Pool

from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm_experiments.online import online_default, online_thetaless

import manager


ITERS_COUNT = 200
SAMPLES = 100


if __name__ == '__main__':
    train_n_dw_matrix, test_n_dw_matrix = main_cases.get_nips(
        train_proportion=0.8
    )[:2]

    args_list = list()
    for T in [20, 50]:
        for phi_alpha in [-0.1, 0., 0.1]:
            for theta_alpha in [-0.1, 0., 0.1]:
                regularization_list = [
                    regularizers.Additive(phi_alpha, theta_alpha)] * ITERS_COUNT
                for batch_size in [100, 500, 1000]:
                    args_list.append((
                        train_n_dw_matrix, test_n_dw_matrix,
                        online_default.Optimizer(
                            regularization_list,
                            sampling=batch_size,
                            calc_global_theta=True,
                        ),
                        T,
                        SAMPLES,
                        '20news_experiment/20news_{}t_online_default_b{}_{}_{}.pkl'.format(
                            T, batch_size, phi_alpha, theta_alpha
                        )
                    ))
                    args_list.append((
                        train_n_dw_matrix, test_n_dw_matrix,
                        online_thetaless.Optimizer(
                            regularization_list,
                            sampling=batch_size,
                            calc_global_theta=True,
                        ),
                        T,
                        SAMPLES,
                        '20news_experiment/20news_{}t_online_thetaless_b{}_{}_{}.pkl'.format(
                            T, batch_size, phi_alpha, theta_alpha
                        )
                    ))

    Pool(processes=5).map(manager.perform_experiment, args_list)
