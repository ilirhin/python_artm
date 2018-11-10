import numpy as np

from artm import common
from artm import EPS
from artm.common import callbacks


def default_callback(
    train_n_dw_matrix=None,
    test_n_dw_matrix=None,
    top_pmi_sizes=None,
    top_avg_jacard_sizes=None,
    uniqueness_measures=False,
    measure_time=False
):
    builder = callbacks.Builder(measure_time=measure_time) \
        .sparsity() \
        .theta_sparsity() \
        .kernel_avg_size() \
        .kernel_avg_jacard() \
        .topic_correlation()
    n_dw_matrix = 0.
    if train_n_dw_matrix is not None:
        builder = builder.perplexity('train_perplexity', train_n_dw_matrix)
        n_dw_matrix += train_n_dw_matrix
    if test_n_dw_matrix is not None:
        builder = builder.perplexity('test_perplexity', test_n_dw_matrix)
        n_dw_matrix += test_n_dw_matrix
    if top_pmi_sizes is not None:
        occurrences, co_occurrences = common.calc_doc_occurrences(n_dw_matrix)
        builder = builder.top_pmi(
            occurrences, co_occurrences,
            n_dw_matrix.shape[0], top_pmi_sizes
        )
    if top_avg_jacard_sizes is not None:
        for top_size in top_avg_jacard_sizes:
            builder = builder.top_avg_jacard(top_size)
    if uniqueness_measures:
        builder = builder.uniqueness_measure()

    return builder.build()


def default_sample(
        train_n_dw_matrix, T, seed, optimizer,
        init_phi_zeros=None, init_theta_zeros=None
):
    D, W = train_n_dw_matrix.shape
    random_gen = np.random.RandomState(seed)
    phi_matrix = common.get_prob_matrix_by_counters(
        random_gen.uniform(size=(T, W)).astype(np.float64)
    )
    if init_phi_zeros is not None:
        phi_matrix = common.get_prob_matrix_by_counters(
            phi_matrix * (init_phi_zeros > EPS)
        )
    theta_matrix = common.get_prob_matrix_by_counters(
        np.ones(shape=(D, T)).astype(np.float64)
    )
    if init_theta_zeros is not None:
        theta_matrix = common.get_prob_matrix_by_counters(
            theta_matrix * (init_theta_zeros > EPS)
        )
    optimizer.iteration_callback.start_launch()
    result = optimizer.run(train_n_dw_matrix, phi_matrix, theta_matrix)
    optimizer.iteration_callback.finish_launch()
    return result
