import numpy as np
from artm.common import get_prob_matrix_by_counters


class Regularizer(object):
    def __init__(self, tau, use_old_phi=False):
        self.tau = tau
        self.use_old_phi = use_old_phi

    def __call__(self, phi, theta, n_tw, n_dt):
        T, _ = phi.shape
        if not self.use_old_phi:
            phi_matrix = get_prob_matrix_by_counters(n_tw)
        else:
            phi_matrix = phi
        aggr_phi = np.sum(phi_matrix, axis=0)
        coeff = - 1. / (T - 1) / T * self.tau
        return (
            coeff * phi_matrix * (aggr_phi - phi_matrix),
            np.zeros_like(n_dt)
        )
