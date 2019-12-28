from __future__ import print_function

from collections import Counter

import numpy as np
from scipy import sparse

from pyartm import common
from pyartm import regularizers
from pyartm.optimizations import default
from pyartm.optimizations import thetaless
from pyartm.optimizations import naive_thetaless


if __name__ == '__main__':
    train_n_dw_matrix = sparse.csr_matrix(np.array([
        [1, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
    ]))
    regularization_list = [regularizers.Trivial()] * 50
    cnt = Counter()
    for iteration in range(1000):
        D, W = train_n_dw_matrix.shape
        T = 4
        random_gen = np.random.RandomState(iteration)
        init_phi_matrix = common.get_prob_matrix_by_counters(
            random_gen.uniform(size=(T, W)).astype(np.float64)
        )
        init_theta_matrix = common.get_prob_matrix_by_counters(
            np.ones(shape=(D, T)).astype(np.float64)
        )

        for module in [default, thetaless, naive_thetaless]:
            optimizer = module.Optimizer(regularization_list, verbose=False)
            phi_matrix, theta_matrix = optimizer.run(
                train_n_dw_matrix, init_phi_matrix, init_theta_matrix
            )
            phi_matrix.sort(axis=0)
            is_good = ((phi_matrix > 1 - 1e-3).astype(np.int) == np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ])).all()
            cnt[module.__name__] += is_good

        if iteration % 100 == 0:
            print(cnt)
