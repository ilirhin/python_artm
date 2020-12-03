from multiprocessing import Pool

from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import obd
from pyartm.optimizations import naive_obd
from pyartm.optimizations import default

import manager


ITERS_COUNT = 100
SAMPLES = 100
INIT_ITERS = 100


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
        plsa_list = [
            regularizers.Trivial()
        ] * ITERS_COUNT
        sparse_lda_list = [
            regularizers.Additive(-1, 0.)
        ] * ITERS_COUNT

        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            default.Optimizer(sparse_lda_list), T, SAMPLES, INIT_ITERS,
            '20news_experiment/20news_{}t_post_lda.pkl'.format(T)
        ))
        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            obd.Optimizer(
                plsa_list,
                gamma_tw_min_delta=1,
            ), T, SAMPLES, INIT_ITERS,
            '20news_experiment/20news_{}t_post_obd_limited.pkl'.format(T)
        ))
        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            naive_obd.Optimizer(
                plsa_list,
                gamma_tw_min_delta=1,
            ), T, SAMPLES, INIT_ITERS,
            '20news_experiment/20news_{}t_post_naive_obd_limited.pkl'.format(T)
        ))
        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            obd.Optimizer(
                plsa_list,
                gamma_tw_min_delta=1,
                gamma_tw_max_delta=10,
                gamma_tw_delta_percentile=0.5,
            ), T, SAMPLES, INIT_ITERS,
            '20news_experiment/20news_{}t_post_obd.pkl'.format(T)
        ))

        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            default.Optimizer(sparse_lda_list), T, SAMPLES, 10,
            '20news_experiment/20news_{}t_pre_lda.pkl'.format(T)
        ))
        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            obd.Optimizer(
                plsa_list,
                gamma_tw_min_delta=1,
            ), T, SAMPLES, 10,
            '20news_experiment/20news_{}t_pre_obd_limited.pkl'.format(T)
        ))
        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            naive_obd.Optimizer(
                plsa_list,
                gamma_tw_min_delta=1,
            ), T, SAMPLES, 10,
            '20news_experiment/20news_{}t_pre_naive_obd_limited.pkl'.format(T)
        ))
        args_list.append((
            train_n_dw_matrix, test_n_dw_matrix,
            obd.Optimizer(
                plsa_list,
                gamma_tw_min_delta=1,
                gamma_tw_max_delta=10,
                gamma_tw_delta_percentile=0.5,
            ), T, SAMPLES, 10,
            '20news_experiment/20news_{}t_pre_obd.pkl'.format(T)
        ))

    Pool(processes=5).map(manager.perform_experiment, args_list)
