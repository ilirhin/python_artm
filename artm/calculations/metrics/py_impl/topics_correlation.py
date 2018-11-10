# coding: utf-8

import numpy as np


def calc_topics_correlation(phi):
    T, W = phi.shape
    unnormed = np.sum(np.sum(phi, axis=0) ** 2) - np.sum(phi ** 2)
    return unnormed / (T * (T - 1))
