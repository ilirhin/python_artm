# coding: utf-8
from multiprocessing import Pool

from artm.datasets import main_cases
from artm import regularizers
from artm.common import experiments
from artm.optimizations import default


def perform_experiment((optimizer, T, samples, output_path)):
    (
        train_n_dw_matrix,
        test_n_dw_matrix,
        _,
        _,
        doc_targets
    ) = main_cases.get_20newsgroups([
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ], train_test_split=0.8)
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix,
        uniqueness_measures=True
    )
    for seed in range(samples):
        print(seed)
        experiments.default_sample(train_n_dw_matrix, T, seed, optimizer)
    optimizer.iteration_callback.save_results(output_path)


if __name__ == '__main__':
    args_list = list()
    phi_alpha = -0.1
    for T in range(3, 16):
        for theta_alpha in [-0.1, 0., 0.1]:
            regularization_list = [
                regularizers.Additive(phi_alpha, theta_alpha)] * 100
            args_list.append((
                default.Optimizer(regularization_list), T, 10,
                'iter_exp/iter_exp_20news_{}t_{}_{}.pkl'.format(
                    T, phi_alpha, theta_alpha
                )
            ))

    Pool(processes=5).map(perform_experiment, args_list)
