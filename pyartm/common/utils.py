from future.builtins import range
from future.utils import iteritems

import numpy as np
import scipy.sparse

from pyartm import EPS


def get_docptr(n_dw_matrix):
    """
    :param n_dw_matrix:
    :return: row indices for the provided matrix
    """
    D, W = n_dw_matrix.shape
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in range(D):
        docptr.extend(
            [doc_num] * (indptr[doc_num + 1] - indptr[doc_num])
        )
    return np.array(docptr, dtype=np.int32)


def get_prob_matrix_by_counters(counters, inplace=False):
    """
    :param counters: matrix to normalize rows
    :param inplace: flag to inplace normalization
    :return:
    """
    if inplace:
        res = counters
    else:
        res = np.copy(counters)
    res[res < 0] = 0.
    # set rows where sum of row is small to uniform
    res[np.sum(res, axis=1) < EPS, :] = 1.
    res /= np.sum(res, axis=1)[:, np.newaxis]
    return res


def calc_doc_occurrences(n_dw_matrix):
    """
    :param n_dw_matrix: sparse document-word matrix, shape is D x W
    :return: sparse matrix of co-occurrences

    doc_occurrences[w1, w2] = the number of the documents
    where there are w1 and w2
    """
    matrix = (scipy.sparse.csc_matrix(n_dw_matrix) > 0).astype(int)
    co_occurrences = matrix.T * matrix
    return co_occurrences.diagonal(), co_occurrences


def pairwise_counters_2_sparse_matrix(co_occurrences):
    """
    :param co_occurrences: dict of co-occurrences

    co_occurrences[(w1, w2)] = the number of the co-occurrences of w1 and w2

    :return: sparse matrix of co_occurrences
    """
    row = []
    col = []
    data = []
    for (w1, w2), value in iteritems(co_occurrences):
        row.append(w1)
        col.append(w2)
        data.append(value)
    return scipy.sparse.csr_matrix((data, (row, col)))
