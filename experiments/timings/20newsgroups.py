from artm import regularizers
from artm.datasets import main_cases
from artm.optimizations import timed_default

import manager


if __name__ == '__main__':
    n_dw_matrix = main_cases.get_20newsgroups([
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ])[0]
    manager.perform_experiment(
        n_dw_matrix, timed_default.Optimizer(
            regularization_list=[regularizers.Additive(0., 0.)] * 100,
            return_counters=True
        ), 10, 100
    )
