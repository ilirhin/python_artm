import numpy as np


class Regularizer(object):
    def __init__(self, *regularizers):
        assert len(regularizers) > 0
        self.regularizers = regularizers

    def __call__(self, phi, theta, n_tw, n_dt):
        r_tw, r_dt = np.zeros_like(n_tw), np.zeros_like(n_dt)
        for regularizer in self.regularizers:
            new_r_tw, new_r_dt = regularizer(phi, theta, n_tw, n_dt)
            r_tw += new_r_tw
            r_dt += new_r_dt
        return r_tw, r_dt
