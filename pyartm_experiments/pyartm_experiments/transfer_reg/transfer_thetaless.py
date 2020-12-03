from future.builtins import range

import numpy as np
import scipy.sparse

from pyartm import EPS
from pyartm import common
from pyartm.optimizations import base


class Optimizer(base.Optimizer):

    def __init__(
        self,
        regularization_list=None,
        loss_function=None,
        return_counters=False,
        const_phi=False,
        inplace=False,
        verbose=True,
        iteration_callback=None,
    ):
        super(Optimizer, self).__init__(
            regularization_list=regularization_list,
            loss_function=loss_function,
            return_counters=return_counters,
            const_phi=const_phi,
            const_theta=False,
            inplace=inplace,
            verbose=verbose,
            iteration_callback=iteration_callback
        )

    def calc_docsizes(self, n_dw_matrix):
        D, _ = n_dw_matrix.shape
        docsizes = []
        indptr = n_dw_matrix.indptr
        for doc_num in range(D):
            size = indptr[doc_num + 1] - indptr[doc_num]
            value = np.sum(
                n_dw_matrix.data[indptr[doc_num]:indptr[doc_num + 1]]
            )
            docsizes.extend([value] * size)
        return np.array(docsizes)

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        n_tw, n_dt = None, None
        docsizes = self.calc_docsizes(n_dw_matrix)

        B = scipy.sparse.csr_matrix(
            (
                1. * n_dw_matrix.data / docsizes,
                n_dw_matrix.indices,
                n_dw_matrix.indptr
            ),
            shape=n_dw_matrix.shape
        ).tocsc()

        for it in range(self.iters_count):
            phi_matrix_tr = np.transpose(phi_matrix)
            phi_rev_matrix = common.get_prob_matrix_by_counters(phi_matrix_tr)

            A = self.calc_A_matrix(
                n_dw_matrix, theta_matrix, docptr,
                phi_matrix_tr, wordptr
            )
            n_dt = A.dot(phi_matrix_tr) * theta_matrix
            n_tw = A.tocsc().T.dot(theta_matrix).T * phi_matrix

            r_tw, r_dt = self.calc_reg_coeffs(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )

            theta_indices = theta_matrix > EPS
            r_dt[theta_indices] /= theta_matrix[theta_indices]
            r_dt[~theta_indices] = 0.
            r_dt = np.clip(r_dt, -100., 100.)
            tmp = r_dt.T * B / (phi_matrix_tr.sum(axis=1) + EPS)
            r_tw += (tmp - np.einsum('ij,ji->i', phi_rev_matrix, tmp)) * phi_matrix
            n_tw += r_tw

            phi_matrix, theta_matrix = self.finish_iteration(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )

        return phi_matrix, theta_matrix, n_tw, n_dt
