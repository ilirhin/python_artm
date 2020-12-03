from multiprocessing import Pool

from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import default, thetaless
from pyartm_experiments.transfer_reg import transfer_thetaless

import manager


ITERS_COUNT = 100
SAMPLES = 5


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
    for T in [10, 25]:
        for theta_alpha in [0.1, 0.01, 0.1]:
            regularization_list = [
                regularizers.Additive(0, theta_alpha)] * ITERS_COUNT
            args_list.append((
                train_n_dw_matrix, test_n_dw_matrix,
                default.Optimizer(regularization_list),
                T, SAMPLES,
                '20news_experiment/20news_{}t_default_{}_{}.pkl'.format(
                    T, 0., theta_alpha
                )
            ))
            args_list.append((
                train_n_dw_matrix, test_n_dw_matrix,
                thetaless.Optimizer(regularization_list),
                T, SAMPLES,
                '20news_experiment/20news_{}t_thetaless_{}_{}.pkl'.format(
                    T, 0., theta_alpha
                )
            ))
            args_list.append((
                train_n_dw_matrix, test_n_dw_matrix,
                transfer_thetaless.Optimizer(regularization_list),
                T, SAMPLES,
                '20news_experiment/20news_{}t_transfer_thetaless_{}_{}.pkl'.format(
                    T, 0., theta_alpha
                )
            ))

    #manager.perform_experiment(args_list[0])
    #manager.perform_experiment(args_list[1])
    manager.perform_experiment(args_list[2])
    #Pool(processes=5).map(manager.perform_experiment, args_list)
