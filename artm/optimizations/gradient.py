import numpy as np

from . import base


class Optimizer(base.Optimizer):
    __slots__ = ('learning_rate',)

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
        learning_rate=1.
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
        self.learning_rate = learning_rate

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        for it in range(self.iters_count):
            phi_matrix_tr = np.transpose(phi_matrix)

            A = self.calc_A_matrix(
                n_dw_matrix, theta_matrix, docptr,
                phi_matrix_tr, wordptr
            ).tocsc()

            g_tw = theta_matrix.T * A
            g_dt = A.dot(phi_matrix_tr)

            r_tw, r_dt = self.calc_reg_coeffs(
                it, phi_matrix, theta_matrix, phi_matrix, theta_matrix
            )
            g_tw += r_tw
            g_dt += r_dt

            g_tw -= np.sum(g_tw * phi_matrix, axis=1)[:, np.newaxis]
            g_dt -= np.sum(g_dt * theta_matrix, axis=1)[:, np.newaxis]

            g_tw *= self.learning_rate
            g_tw += phi_matrix

            g_dt *= self.learning_rate
            g_dt += theta_matrix

            phi_matrix, theta_matrix = self.finish_iteration(
                it, phi_matrix, theta_matrix, g_tw, g_dt
            )

        return phi_matrix, theta_matrix, None, None
