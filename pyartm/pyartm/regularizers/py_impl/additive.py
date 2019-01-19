import numpy as np


class Regularizer(object):
    def __init__(self, phi_alpha, theta_alpha):
        self.phi_alpha = phi_alpha
        self.theta_alpha = theta_alpha

    def __call__(self, phi, theta, n_tw, n_dt):
        return (
            np.zeros_like(n_tw) + self.phi_alpha,
            np.zeros_like(n_dt) + self.theta_alpha
        )
