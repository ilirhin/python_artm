from __future__ import print_function

import pickle
import os

import numpy as np
import scipy.sparse

from pyartm.datasets import main_cases
from pyartm import regularizers
from pyartm.common import experiments
from pyartm.calculations import metrics
from pyartm.optimizations import default


def get_optimizer(phi_alpha, iters_count):
    return default.Optimizer(
        [regularizers.Additive(phi_alpha, 0.)] * iters_count
    )


def perform_lda(
    n_dw_matrix, optimizer, T, samples,
    output_path, init_phi=None, init_theta=None
):
    calc_perplexity = metrics.calc_perplexity_function(n_dw_matrix)
    phis = list()
    perplexities = list()
    for seed in range(samples):
        print(seed)
        phi, theta = experiments.default_sample(
            n_dw_matrix, T, seed, optimizer,
            init_phi_zeros=init_phi, init_theta_zeros=init_theta
        )
        phis.append(phi.flatten())
        perplexities.append(calc_perplexity(phi, theta))
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as output_file:
        pickle.dump({
            'init_phi': init_phi,
            'init_theta': init_theta,
            'perplexities': perplexities,
            'phis': phis
        }, output_file)


if __name__ == '__main__':
    n_dw_matrix, _, _, _ = main_cases.get_20newsgroups([
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ])

    print('Original PLSA')
    perform_lda(
        n_dw_matrix, optimizer=get_optimizer(0., 100), T=10,
        samples=300, output_path='stability_exp/plsa.pkl'
    )

    print('Full initialized PLSA')
    phi, theta = experiments.default_sample(
        n_dw_matrix, T=10, seed=42, optimizer=get_optimizer(-0.1, 100)
    )
    init_phi, init_theta = experiments.default_sample(
        n_dw_matrix, T=10, seed=42, optimizer=get_optimizer(0., 100),
        init_phi_zeros=phi, init_theta_zeros=theta
    )
    perform_lda(
        n_dw_matrix, optimizer=get_optimizer(0., 100), T=10,
        samples=300, output_path='stability_exp/full_initialized_plsa.pkl',
        init_phi=init_phi, init_theta=init_theta
    )

    print('Synthetic PLSA')
    matrix = np.dot(init_theta, init_phi)
    matrix[np.isnan(matrix)] = 0.
    synthetic_n_dw_matrix = scipy.sparse.csr_matrix(matrix)
    synthetic_n_dw_matrix.eliminate_zeros()

    perform_lda(
        synthetic_n_dw_matrix, optimizer=get_optimizer(0., 100), T=10,
        samples=100, output_path='stability_exp/synthetic_plsa.pkl'
    )

    print('Full initialized synthetic PLSA')
    perform_lda(
        synthetic_n_dw_matrix, optimizer=get_optimizer(0., 100), T=10,
        samples=100, output_path='stability_exp/full_initialized_synthetic_plsa.pkl',
        init_phi=init_phi, init_theta=init_theta
    )
