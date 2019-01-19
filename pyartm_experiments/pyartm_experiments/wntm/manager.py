from __future__ import print_function

import os

import numpy as np

from pyartm import common

from pyartm.calculations import metrics
from pyartm import EPS
from pyartm.common import experiments
from pyartm.common import callbacks

from pyartm.common.timers import SimpleTimer


def symmetric_sample(train_n_dw_matrix, T, seed, optimizer):
    D, W = train_n_dw_matrix.shape
    if D != W:
        raise ValueError('D must be equal to W in symmetric_sample')
    random_gen = np.random.RandomState(seed)
    phi_matrix = common.get_prob_matrix_by_counters(
        random_gen.uniform(size=(T, W)).astype(np.float64)
    )
    theta_matrix = common.get_prob_matrix_by_counters(phi_matrix.T)
    optimizer.iteration_callback.start_launch()
    result = optimizer.run(train_n_dw_matrix, phi_matrix, theta_matrix)
    optimizer.iteration_callback.finish_launch()
    return result


class Callback(callbacks.Basic):
    def __init__(self, n_dw_matrix):
        self.calc_perplexity = metrics.calc_perplexity_function(n_dw_matrix)

    def __call__(self, it, phi, theta):
        print(it, 'iteration:')
        with SimpleTimer('callback stats'):
            if it % 10 == 0:
                print('\tperplexity:', self.calc_perplexity(phi, theta))
            print('\tphi_sparsity:', 1. * np.sum(phi < EPS) / np.sum(phi >= EPS))
            print('\ttheta_sparsity:', 1. * np.sum(theta < EPS) / np.sum(theta >= EPS))
        print('\t{}'.format(SimpleTimer.total_times))
        print()


def perform_ww_experiment((n_ww_matrix, optimizer, T, samples, output_dir)):
    optimizer.iteration_callback = Callback(n_ww_matrix)
    for seed in range(samples):
        print('Seed', seed)
        seed_callback = experiments.default_callback(
            train_n_dw_matrix=n_ww_matrix,
            top_avg_jaccard_sizes=[10, 50, 100, 200]
        )
        phi, theta, n_tw, n_dt = symmetric_sample(
            n_ww_matrix, T, seed, optimizer
        )
        seed_callback.start_launch()
        seed_callback(0, phi, theta)
        seed_callback.finish_launch()
        result = dict(phi=phi, theta=theta, n_tw=n_tw, n_dt=n_dt)
        result['properties'] = {
            key: value[0][0]
            for key, value in seed_callback.result.items()
        }
        callbacks.save_results(
            result,
            os.path.join(output_dir, 'seed_{}.pkl'.format(seed))
        )
