import numpy as np
import scipy.sparse

from artm import EPS
from artm import range
from artm import common
from . import base


class Optimizer(base.Optimizer):
    __slots__ = ('use_B_cheat',)

    def __init__(
        self,
        regularization_list=None,
        loss_function=None,
        return_counters=False,
        const_phi=False,
        inplace=False,
        verbose=True,
        iteration_callback=None,
        use_B_cheat=False
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
        self.use_B_cheat = use_B_cheat

    def calc_docsizes(self, n_dw_matrix):
        D, _ = n_dw_matrix.shape
        docsizes = []
        indptr = n_dw_matrix.indptr
        for doc_num in range(D):
            size = indptr[doc_num + 1] - indptr[doc_num]
            if self.use_B_cheat:
                docsizes.extend([size] * size)
            else:
                value = np.sum(
                    n_dw_matrix.data[indptr[doc_num]:indptr[doc_num + 1]]
                )
                docsizes.extend([value] * size)
        return np.array(docsizes)

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        n_tw = None
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
            theta_matrix = common.get_prob_matrix_by_counters(
                n_dw_matrix.dot(phi_rev_matrix)
            )

            A = self.calc_A_matrix(
                n_dw_matrix, theta_matrix, docptr,
                phi_matrix_tr, wordptr
            ).tocsc()

            n_tw = A.T.dot(theta_matrix).T * phi_matrix
            r_tw, r_dt = self.calc_reg_coeffs(
                it, phi_matrix, theta_matrix, n_tw, theta_matrix
            )

            theta_indices = theta_matrix > EPS
            r_dt[theta_indices] /= theta_matrix[theta_indices]
            r_dt[~theta_indices] = 0.

            g_dt = A.dot(phi_matrix_tr) + r_dt
            tmp = g_dt.T * B / (phi_matrix_tr.sum(axis=1) + EPS)
            r_tw += (tmp - np.einsum('ij,ji->i', phi_rev_matrix, tmp)) * phi_matrix
            n_tw += r_tw

            phi_matrix, theta_matrix = self.finish_iteration(
                it, phi_matrix, theta_matrix, n_tw, None
            )

        return phi_matrix, theta_matrix, n_tw, None
