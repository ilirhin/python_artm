from future.builtins import range

import numpy as np


def create_pmi_top_function(
    doc_occurrences, doc_co_occurrences,
    documents_number, top_sizes,
    co_occurrences_smooth=1.
):
    """
    :param doc_occurrences: array of doc occurrences of words
    :param doc_co_occurrences: sparse matrix of doc co-occurrences of words
    :param documents_number: number of the documents
    :param top_sizes: list of top values to calculate top-pmi for
    :param co_occurrences_smooth: constant to smooth co-occurrences in log
    :return: function which takes phi and theta and returns
    pair of two arrays: pmi-s of the tops and ppmi-s of the tops

    pmi[i] - pmi(top of size top_sizes[i])
    ppmi[i] - ppmi(top of size top_sizes[i])

    pmi(words) = sum_{u in words, v in words, u != v}
    log(
        (doc_co_occurrences[u, v] * documents_number + co_occurrences_smooth)
        / doc_occurrences[u] / doc_occurrences[v]
    )

    ppmi(words) = sum_{u in words, v in words, u != v}
    max(log(
        (doc_co_occurrences[u, v] * documents_number + co_occurrences_smooth)
        / doc_occurrences[u] / doc_occurrences[v]
    ), 0)

    """
    def func(phi):
        T, W = phi.shape
        max_top_size = max(top_sizes)
        pmi, ppmi = np.zeros(max_top_size), np.zeros(max_top_size)
        tops = np.argpartition(phi, -max_top_size, axis=1)[:, -max_top_size:]
        for t in range(T):
            top = sorted(tops[t], key=lambda w: - phi[t, w])
            co_occurrences = doc_co_occurrences[top, :][:, top].todense()
            occurrences = doc_occurrences[top]
            values = np.log(
                (co_occurrences * documents_number + co_occurrences_smooth)
                / occurrences[:, np.newaxis]
                / occurrences[np.newaxis, :]
            )
            diag = np.diag_indices(len(values))
            values.cumsum(axis=0).cumsum(axis=1)[diag] - values[diag].cumsum()
            pmi += np.array(
               values.cumsum(axis=0).cumsum(axis=1)[diag] - values[diag].cumsum()
            ).ravel()
            values[values < 0.] = 0.
            ppmi += np.array(
               values.cumsum(axis=0).cumsum(axis=1)[diag] - values[diag].cumsum()
            ).ravel()
        sizes = np.arange(2, max_top_size + 1)
        pmi[1:] /= (T * sizes * (sizes - 1))
        ppmi[1:] /= (T * sizes * (sizes - 1))
        indices = np.array(top_sizes) - 1
        return pmi[indices], ppmi[indices]

    return func
