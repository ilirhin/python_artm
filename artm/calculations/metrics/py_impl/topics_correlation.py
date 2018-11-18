import numpy as np


def calc_topics_correlation(phi):
    """
    :param phi: topics-words matrix, shape T x W, stochastic over W
    :return:
    1. / T(T-1) * sum_{w, t1, t2, t1 != t2} phi_{t1, w} * phi_{t2, w} =
    1. / T(T-1) * (sum_{w, t1, t2} phi_{t1, w} * phi_{t2, 2} - sum_{w, t} phi_{t, w} ** 2 =
    1. / T(T-1) * (sum_{w}  (sum_t phi_{t, w}) ** 2 - sum_{w, t} phi_{t, w} ** 2
    """
    T, W = phi.shape
    unnormed = np.sum(np.sum(phi, axis=0) ** 2) - np.sum(phi ** 2)
    return unnormed / (T * (T - 1))
