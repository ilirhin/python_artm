from multiprocessing import Pool

from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import obd
from pyartm.optimizations import default

import manager


ITERS_COUNT = 100
SAMPLES = 100


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

    args_list = list()
    for T in [10, 25]:
        args_list.append((
            train_n_dw_matrix, T,
            '20news_experiment/20news_{}t_plots.pkl'.format(T)
        ))

    Pool(processes=5).map(manager.perform_plots, args_list)
