import numpy as np

from artm import common
from artm import EPS
from artm.common import callbacks


def default_callback(
    train_n_dw_matrix=None,
    test_n_dw_matrix=None,
    top_pmi_sizes=None,
    top_avg_jaccard_sizes=None,
    uniqueness_measures=False,
    measure_time=False,
    forced_occurrences_co_occurrences_tuple=None,
    collect_phi=False,
    collect_theta=False
):
    """
    :param train_n_dw_matrix:
    :param test_n_dw_matrix:
    :param top_pmi_sizes: sizes of tops to calc pmi for
    :param top_avg_jaccard_sizes: sizes of tops to calc avg_pairwise_jaccard for
    :param uniqueness_measures: flag to calc uniqueness_measures
    :param measure_time: flag to calc measure time of the parts
    :param forced_occurrences_co_occurrences_tuple: tuple of occurrences and co-occurrences
    :param collect_phi: collect phis over iterations
    :param collect_theta: collect thetas over iterations
    :return:
    """
    builder = callbacks.Builder(measure_time=measure_time) \
        .sparsity() \
        .theta_sparsity() \
        .kernel_avg_size() \
        .kernel_avg_jaccard() \
        .topic_correlation()
    n_dw_matrix = 0.
    if train_n_dw_matrix is not None:
        builder = builder.perplexity('train_perplexity', train_n_dw_matrix)
        n_dw_matrix += train_n_dw_matrix
    if test_n_dw_matrix is not None:
        builder = builder.perplexity('test_perplexity', test_n_dw_matrix)
        n_dw_matrix += test_n_dw_matrix
    if top_pmi_sizes is not None:
        if forced_occurrences_co_occurrences_tuple is None:
            occurrences, co_occurrences = common.calc_doc_occurrences(n_dw_matrix)
        else:
            occurrences, co_occurrences = forced_occurrences_co_occurrences_tuple
        builder = builder.top_pmi(
            occurrences, co_occurrences,
            n_dw_matrix.shape[0], top_pmi_sizes
        )
    if top_avg_jaccard_sizes is not None:
        for top_size in top_avg_jaccard_sizes:
            builder = builder.top_avg_jaccard(top_size)
    if uniqueness_measures:
        builder = builder.uniqueness_measure()
    if collect_phi:
        builder = builder.phi()
    if collect_theta:
        builder = builder.theta()
    return builder.build()


def default_sample(
        train_n_dw_matrix, T, seed, optimizer,
        init_phi_zeros=None, init_theta_zeros=None
):
    """
    :param train_n_dw_matrix:
    :param T: number of topics
    :param seed: seed for random
    :param optimizer: artm.optimizations.base.Optimizer
    :param init_phi_zeros: matrix to init zeros for phi
    :param init_theta_zeros: matrix to init zeros for theta
    :return:
    """
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
