import numpy as np
from numba import jit

from . import base


@jit(nopython=True)
def eval_alpha_tw(
        n_dw_data, s_dw_data,
        theta_matrix, docptr,
        phi_matrix_tr, wordptr
):
    W, T = phi_matrix_tr.shape
    D, _ = theta_matrix.shape
    size = len(docptr)
    result = np.zeros(shape=(W, T))
    for i in range(size):
        d = docptr[i]
        w = wordptr[i]
        n_dw = n_dw_data[i]
        s_dw = s_dw_data[i]
        for t in range(T):
            p_tdw = phi_matrix_tr[w, t] * theta_matrix[d, t] * s_dw
            result[w, t] += n_dw * np.log(1 - p_tdw)
    return result.T


@jit(nopython=True)
def eval_beta_tw(
        n_dw_data, s_dw_data,
        theta_matrix, docptr,
        phi_matrix_tr, wordptr
):
    W, T = phi_matrix_tr.shape
    D, _ = theta_matrix.shape
    size = len(docptr)
    result = np.zeros(shape=(W, T))
    for i in range(size):
        d = docptr[i]
        w = wordptr[i]
        n_dw = n_dw_data[i]
        s_dw = s_dw_data[i]
        for t in range(T):
            p_tdw = phi_matrix_tr[w, t] * theta_matrix[d, t] * s_dw
            result[w, t] += n_dw * p_tdw * p_tdw
    return result.T


def eval_gamma_tw(
        n_dw_matrix, s_dw_data,
        theta_matrix, docptr,
        phi_matrix_tr, wordptr,
        n_tw,
):
    alpha_tw = eval_alpha_tw(
        n_dw_matrix.data, s_dw_data,
        theta_matrix, docptr,
        phi_matrix_tr, wordptr
    )
    beta_tw = eval_beta_tw(
        n_dw_matrix.data, s_dw_data,
        theta_matrix, docptr,
        phi_matrix_tr, wordptr
    )
    norm = phi_matrix_tr.T / (1 - phi_matrix_tr.T)
    beta_t = beta_tw.sum(axis=1)
    n_t = n_tw.sum(axis=1)
    return alpha_tw + norm * (
            n_t[:, np.newaxis] - n_tw
            - 0.5 * norm * (beta_t[:, np.newaxis] - beta_tw)
    )


class Optimizer(base.Optimizer):
    __slots__ = (
        'gamma_tw_min_delta',
        'gamma_tw_max_delta',
        'gamma_callback',
        'gamma_tw_delta_percentile',
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
        iteration_callback=None,
        gamma_tw_min_delta=0.5,
        gamma_tw_max_delta=1000000,
        gamma_tw_delta_percentile=0,
        gamma_callback=None,
    ):
        super(Optimizer, self).__init__(
            regularization_list=regularization_list,
            loss_function=loss_function,
            return_counters=return_counters,
            const_phi=const_phi,
            const_theta=const_theta,
            inplace=inplace,
            verbose=verbose,
            iteration_callback=iteration_callback
        )
        self.gamma_tw_min_delta = gamma_tw_min_delta
        self.gamma_tw_max_delta = gamma_tw_max_delta
        self.gamma_callback = gamma_callback
        self.gamma_tw_delta_percentile = gamma_tw_delta_percentile

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        n_tw, n_dt = None, None
        for it in range(self.iters_count):
            phi_matrix_tr = np.transpose(phi_matrix)

            s_data = self.calc_s_data(
                theta_matrix, docptr,
                phi_matrix_tr, wordptr
            )
            A = self.calc_A_matrix(
                n_dw_matrix, theta_matrix, docptr,
                phi_matrix_tr, wordptr, s_data=s_data
            )
            n_dt = A.dot(phi_matrix_tr) * theta_matrix
            n_tw = np.transpose(
                A.tocsc().transpose().dot(theta_matrix)) * phi_matrix

            gamma_tw = eval_gamma_tw(
                n_dw_matrix, s_data,
                theta_matrix, docptr,
                phi_matrix_tr, wordptr,
                n_tw,
            )
            if self.gamma_callback:
                self.gamma_callback(it, n_tw, n_dt, gamma_tw, 0)
            threshold = np.percentile(
                gamma_tw[gamma_tw <= - self.gamma_tw_min_delta],
                q=100 - self.gamma_tw_delta_percentile
            )
            n_tw[gamma_tw > max(threshold, -self.gamma_tw_max_delta)] = 0.

            r_tw, r_dt = self.calc_reg_coeffs(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )
            n_tw += r_tw
            n_dt += r_dt

            phi_matrix, theta_matrix = self.finish_iteration(
                it, phi_matrix, theta_matrix, n_tw, n_dt
            )

        return phi_matrix, theta_matrix, n_tw, n_dt
