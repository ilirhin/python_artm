import numpy as np

from pyartm import common

from pyartm.optimizations import base


class Optimizer(base.Optimizer):
    __slots__ = ('sampling',)

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
        sampling=1.,
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
        self.sampling = sampling

    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
        n_tw, n_dt = None, None
        D, _ = theta_matrix.shape
        if isinstance(self.sampling, float):
            sample_count = int(D * self.sampling)
        elif isinstance(self.sampling, int):
            sample_count = self.sampling
        else:
            sample_count = None
        gen = np.random.RandomState(47)
        _n_dw_matrix = n_dw_matrix
        _docptr = docptr
        for it in range(self.iters_count):
            indices = gen.choice(D, sample_count, replace=True)
            n_dw_matrix = _n_dw_matrix[indices, :]
            docptr = common.get_docptr(n_dw_matrix)
            wordptr = n_dw_matrix.indices
            phi_matrix_tr = np.transpose(phi_matrix)
            phi_rev_matrix = common.get_prob_matrix_by_counters(phi_matrix_tr)
            theta_matrix = common.get_prob_matrix_by_counters(
                n_dw_matrix.dot(phi_rev_matrix)
            )

            A = self.calc_A_matrix(
                n_dw_matrix, theta_matrix, docptr,
                phi_matrix_tr, wordptr
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

        phi_matrix_tr = np.transpose(phi_matrix)
        phi_rev_matrix = common.get_prob_matrix_by_counters(phi_matrix_tr)
        theta_matrix = common.get_prob_matrix_by_counters(
            _n_dw_matrix.dot(phi_rev_matrix)
        )
        return phi_matrix, theta_matrix, None, None
