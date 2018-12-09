from __future__ import print_function

import time
import sys

import numpy as np
import scipy.sparse

from pyartm import common
from pyartm import loss_functions as losses
from pyartm.calculations.inner_product import memory_efficient_inner1d


class Optimizer(object):
    __slots__ = (
        'regularization_list', 'loss_function', 'inplace',
        'return_counters', 'const_phi', 'const_theta', 'verbose',
        'iteration_callback'
    )

    def __init__(
        self,
        regularization_list=None,
        loss_function=None,
        return_counters=False,
        const_phi=False,
        const_theta=False,
        inplace=False,
        verbose=True,
        iteration_callback=None
    ):
        if regularization_list is None:
            regularization_list = list()
        if loss_function is None:
            loss_function = losses.LogFunction()
        self.regularization_list = regularization_list
        self.loss_function = loss_function
        self.return_counters = return_counters
        self.const_phi = const_phi
        self.const_theta = const_theta
        self.inplace = inplace
        self.verbose = verbose
        self.iteration_callback = iteration_callback

    @property
    def iters_count(self):
        return len(self.regularization_list)

    def finish_iteration(self, it, phi_matrix, theta_matrix, n_tw, n_dt):
        if not self.const_phi and n_tw is not None:
            phi_matrix = common.get_prob_matrix_by_counters(n_tw)
        if not self.const_theta and n_dt is not None:
            theta_matrix = common.get_prob_matrix_by_counters(n_dt)
        if self.iteration_callback is not None:
            self.iteration_callback(it, phi_matrix, theta_matrix)
        return phi_matrix, theta_matrix

    def calc_s_data(self, theta_matrix, docptr, phi_matrix_tr, wordptr):
        return self.loss_function.calc_der(
            memory_efficient_inner1d(
                theta_matrix, docptr,
                phi_matrix_tr, wordptr
            )
        )

    def calc_A_matrix(
        self, n_dw_matrix, theta_matrix, docptr, phi_matrix_tr, wordptr
    ):
        return scipy.sparse.csr_matrix(
            (
                n_dw_matrix.data * self.calc_s_data(
                    theta_matrix, docptr, phi_matrix_tr, wordptr
                ),
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        )

    def calc_reg_coeffs(self, it, phi_matrix, theta_matrix, n_tw, n_dt):
        return self.regularization_list[it](
            phi_matrix, theta_matrix, n_tw, n_dt
        )

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        """
        :param n_dw_matrix: documents-words matrix D x W
        :param docptr: docptr for n_dw_matrix (for all occurrences)
        :param wordptr: wordptr for n_dw_matrix (for all occurrences)
        :param phi_matrix: matrix T x W, can be modified
        :param theta_matrix: D x T, can be modified
        :return: phi_matrix, theta_matrix, n_tw, n_dt after optimization
        """
        raise NotImplementedError()

    def run(self, n_dw_matrix, init_phi_matrix, init_theta_matrix):
        start = time.time()

        if not self.inplace:
            init_phi_matrix = np.copy(init_phi_matrix)
            init_theta_matrix = np.copy(init_theta_matrix)

        docptr = common.get_docptr(n_dw_matrix)
        wordptr = n_dw_matrix.indices

        phi_matrix, theta_matrix, n_tw, n_dt = self._run(
            n_dw_matrix, docptr, wordptr,
            init_phi_matrix, init_theta_matrix
        )
        if self.verbose:
            print("fit time: {}".format(time.time() - start), file=sys.stderr)

        if self.const_phi and init_phi_matrix != phi_matrix:
            raise ValueError("phi is not constant")

        if self.const_phi and init_theta_matrix != theta_matrix:
            raise ValueError("theta is not constant")

        if self.return_counters:
            return phi_matrix, theta_matrix, n_tw, n_dt
        else:
            return phi_matrix, theta_matrix
