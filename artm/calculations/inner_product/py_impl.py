import numpy as np
from numpy.core.umath_tests import inner1d

MAX_INNER1D_ELEMENTS = 500000000


def memory_efficient_inner1d(
        fst_arr,
        fst_indices,
        snd_arr,
        snd_indices,
        max_stored_elements=MAX_INNER1D_ELEMENTS
):
    assert fst_arr.shape[1] == snd_arr.shape[1]
    assert len(fst_indices) == len(snd_indices)

    _, T = fst_arr.shape
    size = len(fst_indices)
    result = np.zeros(size)
    batch_size = max_stored_elements / T

    start = 0
    while start < size:
        finish = min(start + batch_size, size)
        result[start:finish] = inner1d(
            fst_arr[fst_indices[start:finish], :],
            snd_arr[snd_indices[start:finish], :]
        )
        start = finish

    return result
