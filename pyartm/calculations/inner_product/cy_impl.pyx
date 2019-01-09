import numpy as np

cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def memory_efficient_inner1d(
        np.ndarray[np.float64_t, ndim=2] fst_arr,
        np.ndarray[np.int32_t] fst_indices,
        np.ndarray[np.float64_t, ndim=2] snd_arr,
        np.ndarray[np.int32_t] snd_indices
):
    cdef int size = fst_indices.shape[0]
    cdef int width = fst_arr.shape[1]
    cdef np.ndarray[np.float64_t] result = np.zeros(size, dtype=np.float64)
    cdef int i, j, fst_index, snd_index
    for i in range(size):
        result[i] = 0
        fst_index = fst_indices[i]
        snd_index = snd_indices[i]
        for j in range(width):
            result[i] += fst_arr[fst_index, j] * snd_arr[snd_index, j]

    return result
