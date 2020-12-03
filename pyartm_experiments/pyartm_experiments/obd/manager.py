import numpy as np

from pyartm.common import experiments
from pyartm import regularizers
from pyartm.optimizations import default, obd
from pyartm.common.callbacks import save_results


def perform_experiment((
   train_n_dw_matrix, test_n_dw_matrix, optimizer,
   T, samples, init_iters, output_path
)):
    init_optimizer = default.Optimizer([regularizers.Trivial()] * init_iters)
    callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix
    )
    init_optimizer.iteration_callback = callback
    optimizer.iteration_callback = callback
    for seed in range(samples):
        print(seed)
        plsa_phi, plsa_theta = experiments.default_sample(
            train_n_dw_matrix=train_n_dw_matrix,
            T=T,
            seed=seed,
            optimizer=init_optimizer,
            finish_launch=False,
        )
        optimizer.run(train_n_dw_matrix, plsa_phi, plsa_theta)
        if optimizer.iteration_callback:
            optimizer.iteration_callback.finish_launch()

    optimizer.iteration_callback.save_results(output_path)


def perform_plots((train_n_dw_matrix, T, output_path)):
    matrices = []
    gamma_callback = (
        lambda it, n_tw, n_dt, gamma_tw, gamma_dt:
        matrices.append((it, np.copy(n_tw), np.copy(n_dt), np.copy(gamma_tw), np.copy(gamma_dt)))
        if it % 20 == 0
        else
        None
    )
    init_optimizer = obd.Optimizer(
        [regularizers.Trivial()] * 100,
        gamma_tw_min_delta=-100000,
        gamma_callback=gamma_callback
    )
    post_optimizer = obd.Optimizer(
        [regularizers.Trivial()] * 121,
        gamma_tw_min_delta=0.1,
        gamma_tw_max_delta=40,
        gamma_tw_delta_percentile=0.5,
        gamma_callback=gamma_callback
    )

    plsa_phi, plsa_theta = experiments.default_sample(
        train_n_dw_matrix=train_n_dw_matrix,
        T=T,
        seed=42,
        optimizer=init_optimizer,
    )
    post_optimizer.run(train_n_dw_matrix, plsa_phi, plsa_theta)

    save_results(matrices, output_path)

