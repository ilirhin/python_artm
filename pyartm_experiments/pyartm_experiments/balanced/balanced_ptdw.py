import numpy as np
import scipy.sparse

from pyartm.optimizations import base
from pyartm.loss_functions import LogFunction


class Optimizer(base.Optimizer):
    __slots__ = tuple()

    def __init__(
        self,
        regularization_list=None,
        return_counters=False,
        const_phi=False,
        const_theta=False,
        inplace=False,
        verbose=True,
        iteration_callback=None
    ):
        super(Optimizer, self).__init__(
            regularization_list=regularization_list,
            loss_function=LogFunction(),
            return_counters=return_counters,
            const_phi=const_phi,
            const_theta=const_theta,
            inplace=inplace,
            verbose=verbose,
            iteration_callback=iteration_callback
        )

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        n_tw, n_dt = None, None
        for it in range(self.iters_count):
            phi_matrix_tr = np.transpose(phi_matrix)

            s_data_old = self.calc_s_data(
                theta_matrix, docptr, phi_matrix_tr, wordptr
            )
            A_old = scipy.sparse.csr_matrix(
                (
                    n_dw_matrix.data * s_data_old,
                    n_dw_matrix.indices,
                    n_dw_matrix.indptr
                ),
                shape=n_dw_matrix.shape
            )
            n_t = (A_old.dot(phi_matrix_tr) * theta_matrix).sum(axis=0)
            p_t = n_t / n_t.sum()
            s_data_new = self.calc_s_data(
                theta_matrix / p_t, docptr, phi_matrix_tr, wordptr
            )
            A = scipy.sparse.csr_matrix(
                (
                    n_dw_matrix.data * s_data_old * s_data_old / s_data_new,
                    n_dw_matrix.indices,
                    n_dw_matrix.indptr
                ),
                shape=n_dw_matrix.shape
            )

            n_dt = A.dot(phi_matrix_tr) * theta_matrix
            n_tw = np.transpose(
                A.tocsc().transpose().dot(theta_matrix)) * phi_matrix

            r_tw, r_dt = self.calc_reg_coeffs(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )
            n_tw += r_tw
            n_dt += r_dt

            phi_matrix, theta_matrix = self.finish_iteration(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )

        return phi_matrix, theta_matrix, n_tw, n_dt
