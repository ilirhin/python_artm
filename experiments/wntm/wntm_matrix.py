from multiprocessing import Pool

from artm.optimizations import timed_default
from artm.optimizations import thetaless
from artm.datasets import main_cases
import reg_funcs

import manager


ITERS_COUNT = 60
T = 100
SAMPLES = 10


def create_exp_args(n_ww_matrix, name, reg_func=None, optimizer=None):
    if optimizer is None:
        optimizer = timed_default.Optimizer([reg_func] * ITERS_COUNT)
    return (
        n_ww_matrix, optimizer, T, SAMPLES,
        'results/{}'.format(name)
    )


if __name__ == '__main__':
    n_ww_matrix = main_cases.get_wntm_matrix()
    args_list = [
        create_exp_args(n_ww_matrix, 'plsa', reg_funcs.plsa),
        create_exp_args(n_ww_matrix, 'plsa_honest', reg_funcs.plsa_honest),
        create_exp_args(n_ww_matrix, 'plsa_origin', reg_funcs.plsa_origin),
        create_exp_args(
            n_ww_matrix, 'plsa_semi_honest', reg_funcs.plsa_semi_honest
        ),
        create_exp_args(
            n_ww_matrix, 'tARTM',
            optimizer=thetaless.Optimizer(
                [reg_funcs.trivial] * ITERS_COUNT,
                use_B_cheat=False
            ),
        ),
        create_exp_args(
            n_ww_matrix, 'tARTM_cheat',
            optimizer=thetaless.Optimizer(
                [reg_funcs.trivial] * ITERS_COUNT,
                use_B_cheat=True
            )
        )
    ]

    Pool(processes=3).map(manager.perform_ww_experiment, args_list)
