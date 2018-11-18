from multiprocessing import Pool

from artm.datasets import main_cases
from artm import regularizers
from artm.optimizations import thetaless
from artm.optimizations import default

import manager


ITERS_COUNT = 100
SAMPLES = 20


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
                args_list.append((
                    default.Optimizer(regularization_list), T, SAMPLES,
                    'nips_experiment/NIPS_{}t_base_{}_{}.pkl'.format(
                        T, phi_alpha, theta_alpha
                    )
                ))
                for use_B_cheat in [False, True]:
                    args_list.append((
                        train_n_dw_matrix, test_n_dw_matrix,
                        thetaless.Optimizer(regularization_list, use_B_cheat=use_B_cheat), T, SAMPLES,
                        'nips_experiment/NIPS_{}t_artm_{}_{}_{}.pkl'.format(
                            T, phi_alpha, theta_alpha, use_B_cheat
                        )
                    ))

    Pool(processes=4).map(manager.perform_experiment, args_list)