from __future__ import print_function

import numpy as np
from scipy import sparse

from pyartm import common
from pyartm.calculations import metrics
from pyartm import regularizers
from pyartm.optimizations import default
from pyartm.optimizations import thetaless
from pyartm.optimizations import naive_thetaless
from pyartm.optimizations import obd


if __name__ == '__main__':
    # train_n_dw_matrix = sparse.csr_matrix(np.array([
    #     [1, 1, 1, 0],
    #     [1, 0, 1, 1],
    #     [1, 1, 0, 1],
    # ]))
    train_n_dw_matrix = sparse.csr_matrix(np.random.RandomState(42).uniform(
        0, 1, size=(100, 1000)
    ) < 0.3)
    regularization_list = [regularizers.Trivial()] * 500
    extra_opt = obd.Optimizer([regularizers.Additive(0, 0)]*500, verbose=False)

    for module in [default, thetaless, naive_thetaless]:
        print(module.__name__)
        optimizer = module.Optimizer(regularization_list, verbose=False)
        D, W = train_n_dw_matrix.shape
        T = 4
        random_gen = np.random.RandomState(47)
        phi_matrix = common.get_prob_matrix_by_counters(
            random_gen.uniform(size=(T, W)).astype(np.float64)
        )
        theta_matrix = common.get_prob_matrix_by_counters(
            np.ones(shape=(D, T)).astype(np.float64)
        )
        phi_matrix, theta_matrix = optimizer.run(
            train_n_dw_matrix, phi_matrix, theta_matrix
        )
        mod_phi_matrix, mod_theta_matrix = extra_opt.run(
            train_n_dw_matrix, phi_matrix, theta_matrix
        )
        perplexity = metrics.calc_perplexity_function(train_n_dw_matrix)
        log_likelihood = metrics.create_calculate_likelihood_like_function(
            train_n_dw_matrix
        )
        print('Perplexity (only by phi): {:0.4f} ({:0.4f})'.format(
            perplexity(phi_matrix, theta_matrix),
            perplexity(mod_phi_matrix, mod_theta_matrix)
        ))
        print('Sparsity (only by phi): {:0.4f} ({:0.4f})'.format(
            (phi_matrix > 1e-10).mean(),
            (mod_phi_matrix > 1e-10).mean()
        ))
        # print('Phi:')
        # print(np.round(phi_matrix, 3))
        # print('Theta:')
        # print(np.round(theta_matrix, 3))
        # print()
