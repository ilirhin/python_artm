from builtins import range

import numpy as np
import scipy.sparse

from artm import EPS


def get_docptr(n_dw_matrix):
    D, W = n_dw_matrix.shape
    docptr = []
    indptr = n_dw_matrix.indptr
    for doc_num in range(D):
        docptr.extend(
            [doc_num] * (indptr[doc_num + 1] - indptr[doc_num])
        )
    return np.array(docptr)


def get_prob_matrix_by_counters(counters, inplace=False):
    if inplace:
        res = counters
    else:
        res = np.copy(counters)
    res[res < 0] = 0.
    res[np.sum(res, axis=1) < EPS, :] = 1.
    res /= np.sum(res, axis=1)[:, np.newaxis]
    return res


def calc_doc_occurrences(n_dw_matrix):
    matrix = (scipy.sparse.csc_matrix(n_dw_matrix) > 0).astype(int)
    co_occurrences = matrix.T * matrix
    return co_occurrences.diagonal(), co_occurrences


def pairwise_counters_2_sparse_matrix(cooccurences):
    row = []
    col = []
    data = []
    for (w1, w2), value in cooccurences.iteritems():
        row.append(w1)
        col.append(w2)
        data.append(value)
    return scipy.sparse.csr_matrix((data, (row, col)))
