from __future__ import print_function

from artm import regularizers
from artm.datasets import main_cases
from artm.optimizations import timed_default
from artm.common import experiments


def perform_experiment((optimizer, T, samples)):
    (
        train_n_dw_matrix,
        test_n_dw_matrix,
        _,
        _
    ) = main_cases.get_twitter_sentiment140(
        test_proportion=0.2, min_docs_occurrences=3
    )
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
        print(timed_default.SimpleTimer.total_times)


if __name__ == '__main__':
    perform_experiment((
        timed_default.Optimizer(
            regularization_list=[regularizers.Additive(0., 0.)] * 100,
            return_counters=True
        ), 10, 100
    ))
