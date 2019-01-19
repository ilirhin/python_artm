import numpy as np
from numba import jit


@jit(nopython=True)
def memory_efficient_inner1d(fst_arr, fst_indices, snd_arr, snd_indices):
    """
    :param fst_arr: 2d array, shape is N x T
    :param fst_indices: indices of the rows in fst_arr
    :param snd_arr: 2d array, shape is M x T
    :param snd_indices: indices of the rows in fst_arr
    :param max_stored_elements: max number of the elements stored in memory
    :return: np.array([
        sum(fst_arr[i, k] * snd_arr[j, k] for k in 0..T)
        for i, j in fst_indices, snd_indices
    ])
    """
    assert fst_arr.shape[1] == snd_arr.shape[1]
    assert len(fst_indices) == len(snd_indices)

    _, T = fst_arr.shape
    size = len(fst_indices)
    result = np.zeros(size)
    for i in range(size):
        fst_index = fst_indices[i]
        snd_index = snd_indices[i]
        for j in range(T):
            result[i] += fst_arr[fst_index, j] * snd_arr[snd_index, j]
    return result
