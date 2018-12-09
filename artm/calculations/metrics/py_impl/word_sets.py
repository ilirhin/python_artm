from future.builtins import range

import numpy as np


def calc_jaccard_distance(fst_set, snd_set):
    """
    :param fst_set: set of objects
    :param snd_set: set of objects
    :return: jaccard distance (https://en.wikipedia.org/wiki/Jaccard_index)
    of the provided two sets
    """
    if fst_set or snd_set:
        return 1. * len(fst_set & snd_set) / (len(fst_set | snd_set))
    else:
        return 0.


def get_kernels(phi):
    """
    :param phi: topics-words matrix, shape T x W, stochastic over W
    :return: list of kernels for each topic

    kernel of the topic t is the set of the words w such that phi[t, w] > 1 / W
    """
    T, W = phi.shape
    return [
        set(np.where(phi[t, :] * W > 1)[0])
        for t in range(T)
    ]


def get_top_words(phi, top_size):
    """
    :param phi: topics-words matrix, shape T x W, stochastic over W
    :param top_size: the size of the top
    :return: list of top words for each topic

    top words is the set of words w of the largest top_size phi[t, w]
    of the topic
    """
    return [
        set(values)
        for values in np.argpartition(phi, -top_size, axis=1)[:, -top_size:]
    ]


def calc_kernels_sizes(phi):
    """
    :param phi: topics-words matrix, shape T x W, stochastic over W
    :return: kernel sizes of the topics

    kernel of the topic t is the set of the words w such that phi[t, w] > 1 / W
    """
    return [len(kernel) for kernel in get_kernels(phi)]


def calc_avg_pairwise_jaccards(sets):
    """
    :param sets: list of sets
    :return: average jaccard distance between these sets
    """
    size = len(sets)
    res = 0.
    for i in range(size):
        for j in range(size):
            if i != j:
                res += calc_jaccard_distance(sets[i], sets[j])
    return res / size / (size - 1)


def calc_avg_pairwise_kernels_jaccards(phi):
    """
   :param phi: topics-words matrix, shape T x W, stochastic over W
   :return: average pairwise jaccard distance of the topics' kernels

   kernel of the topic t is the set of the words w such that phi[t,w] > 1 / W
   """
    return calc_avg_pairwise_jaccards(get_kernels(phi))


def calc_avg_top_words_jaccards(phi, top_size):
    """
    :param phi: topics-words matrix, shape T x W, stochastic over W
    :param top_size: the size of the top
    :return: average pairwise jaccard distance of the topics' tops
    """
    return calc_avg_pairwise_jaccards(get_top_words(phi, top_size))
