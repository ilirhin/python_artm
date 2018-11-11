import numpy as np

from artm import common


def plsa(phi, theta, n_tw, n_dt):
    return np.zeros_like(phi), - n_dt + np.transpose(n_tw)


def plsa_honest(phi, theta, n_tw, n_dt):
    n_w = np.sum(n_tw, axis=0)
    phi_matrix = common.get_prob_matrix_by_counters(n_tw)
    theta_matrix = common.get_prob_matrix_by_counters(phi_matrix.T)

    ans = - n_w * theta_matrix.T / 2.
    return ans, - n_dt + np.transpose(n_tw + ans)


def plsa_origin(phi, theta, n_tw, n_dt):
    return np.zeros_like(phi), np.zeros_like(theta)


def plsa_semi_honest(phi, theta, n_tw, n_dt):
    n_w = np.sum(n_tw, axis=0)
    phi_matrix = common.get_prob_matrix_by_counters(n_tw)
    theta_matrix = common.get_prob_matrix_by_counters(phi_matrix.T)

    ans = - 0.25 * n_w * theta_matrix.T / 2.
    return ans, - n_dt + np.transpose(n_tw + ans)


def trivial(phi, theta, n_tw, n_dt):
    return np.zeros_like(phi), np.zeros_like(theta)