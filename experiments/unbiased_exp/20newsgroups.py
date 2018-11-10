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
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ], train_test_split=0.8)
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix,
        top_pmi_sizes=[5, 10, 20, 30],
        top_avg_jacard_sizes=[10, 50, 100, 200],
        measure_time=True
    )
    for seed in range(samples):
        print(seed)
        experiments.default_sample(train_n_dw_matrix, T, seed, optimizer)
    optimizer.iteration_callback.save_results(output_path)


if __name__ == '__main__':
    args_list = list()
    for T in [10, 30]:
        for tau in [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]:
            for use_old_phi in [False, True]:
                regularization_list = [
                    regularizers.Decorrelator(tau, use_old_phi)] * 100
                args_list.append((
                    default.Optimizer(regularization_list), T, 100,
                    '20news_experiment/20news_{}t_{}_{}.pkl'.format(
                        T, int(tau), use_old_phi
                    )
                ))

    Pool(processes=8).map(perform_experiment, args_list)
