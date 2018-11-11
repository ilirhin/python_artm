from __future__ import print_function

from artm.optimizations import timed_default
from artm.common import experiments


def perform_experiment(n_dw_matrix, optimizer, T, samples):
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=n_dw_matrix,
        top_pmi_sizes=[5, 10, 20, 30],
        top_avg_jacard_sizes=[10, 50, 100, 200],
        measure_time=True
    )
    for seed in range(samples):
        print(seed)
        experiments.default_sample(n_dw_matrix, T, seed, optimizer)
        print(timed_default.SimpleTimer.total_times)
