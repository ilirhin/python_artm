import numpy as np

from pyartm import common
from pyartm.calculations.inner_product import memory_efficient_inner1d
from pyartm.loss_functions import LogFunction


def create_calculate_likelihood_like_function(n_dw_matrix, loss_function=None):
    """
    :param n_dw_matrix: sparse document-word matrix, shape is D x W
    :param loss_function: loss function
    :return: function, which takes phi and theta and returns
    sum_{w, d} n_dw * loss_function(sum_t phi_tw * theta_dt)
    """
    if loss_function is None:
        loss_function = LogFunction()

    docptr = common.get_docptr(n_dw_matrix)
    wordptr = n_dw_matrix.indices

    def fun(phi_matrix, theta_matrix):
        s_data = loss_function.calc(memory_efficient_inner1d(
            theta_matrix, docptr,
            np.transpose(phi_matrix), wordptr
        ))
        return np.sum(n_dw_matrix.data * s_data)

    return fun


def calc_perplexity_function(n_dw_matrix):
    """
    :param n_dw_matrix: sparse document-word matrix, shape is D x W
    :return: function which takes phi and theta and returns perplexity
    perplexity is e^{-ln_likelihood / n_dw_matrix.sum()}
    """
    helper = create_calculate_likelihood_like_function(
        loss_function=LogFunction(),
        n_dw_matrix=n_dw_matrix
    )
    total_words_number = n_dw_matrix.sum()
    return lambda phi, theta: np.exp(- helper(phi, theta) / total_words_number)
