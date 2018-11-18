from builtins import range

import numpy as np


def calc_jacard_distance(fst_set, snd_set):
    if fst_set or snd_set:
        return 1. * len(fst_set & snd_set) / (len(fst_set | snd_set))
    else:
        return 0.


def get_kernels(phi):
    T, W = phi.shape
    return [
        set(np.where(phi[t, :] * W > 1)[0])
        for t in range(T)
    ]


def get_top_words(phi, top_size):
    return [
        set(values)
        for values in np.argpartition(phi, -top_size, axis=1)[:, -top_size:]
    ]


def calc_kernels_sizes(phi):
    return [len(kernel) for kernel in get_kernels(phi)]


def calc_avg_pairwise_jacards2(sets):
    size = len(sets)
    res = 0.
    for i in range(size):
        for j in range(size):
            if i != j:
                res += calc_jacard_distance(sets[i], sets[j])
    return res / size / (size - 1)


def calc_avg_pairwise_jacards(sets):
    size = len(sets)
    res = 0.
    for i in range(size):
        for j in range(size):
            if i != j:
                res += calc_jacard_distance(sets[i], sets[j])
    return res / size / (size - 1)


def calc_avg_pairwise_kernels_jacards(phi):
    return calc_avg_pairwise_jacards(get_kernels(phi))


def calc_avg_top_words_jacards(phi, top_size):
    return calc_avg_pairwise_jacards(get_top_words(phi, top_size))
