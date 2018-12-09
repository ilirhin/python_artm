from future.builtins import range

import numpy as np


def calc_phi_uniqueness_measures(phi):
    """
    :param phi: topics-words matrix, shape T x W, stochastic over W
    :return: pair of uniqueness and normalized uniqueness measures arrays
    """
    T, W = phi.shape
    uniqueness_measures = list()
    normalized_uniqueness_measures = list()
    for t in range(T):
        positions = phi[t, :] == 0.
        topics = [x for x in range(T) if x != t]
        if np.sum(positions) == 0:
            uniqueness_measures.append(0.)
            normalized_uniqueness_measures.append(0.)
        else:
            rank = np.linalg.matrix_rank(phi[np.ix_(topics, positions)])
            if rank == T - 1:
                max_val = np.min(np.linalg.svd(phi[topics, :])[1])
                curr_val = np.min(np.linalg.svd(phi[np.ix_(topics, positions)])[1])
                uniqueness_measures.append(curr_val)
                normalized_uniqueness_measures.append(curr_val / max_val)
            else:
                uniqueness_measures.append(0.)
                normalized_uniqueness_measures.append(0.)
    return uniqueness_measures, normalized_uniqueness_measures
