from builtins import range

import numpy as np


def create_pmi_top_function(
    doc_occurrences, doc_co_occurrences,
    documents_number, top_sizes,
    co_occurrences_smooth=1.
):
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
