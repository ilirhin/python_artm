import numpy as np

from pyartm_datasets import main_cases
from pyartm import regularizers

from pyartm.common import experiments
from pyartm.optimizations import thetaless, default


ITERS_COUNT = 100
SAMPLES = 100
T = 30


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

    regularization_list = [regularizers.Additive(0, 0)] * ITERS_COUNT

    results = []
    for seed in range(10):
        print(seed)
        results.append(
            experiments.default_sample(
                train_n_dw_matrix,
                T,
                seed,
                thetaless.Optimizer(regularization_list, use_B_cheat=False)
            )
            # experiments.default_sample(
            #     train_n_dw_matrix,
            #     T,
            #     seed,
            #     default.Optimizer(regularization_list)
            # )
        )

    phi1, theta1 = results[0]
    for t1 in range(T):
        best_dist = None
        for i in range(1, len(results)):
            phi2, theta2 = results[i]
            for t2 in range(T):
                dist = np.abs(phi1[t1, :] - phi2[t2, :]).sum()
                if best_dist is None or dist < best_dist:
                    best_dist = dist
        print(t1, best_dist)



