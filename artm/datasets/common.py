from __future__ import print_function
from future.builtins import range
from future.utils import iteritems

import random
from collections import Counter

import scipy.sparse


def create_sparse_matrices(
        documents, train_proportion=None,
        process_log_step=None, random_seed=42
):
    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    random_gen = random.Random(random_seed)
    max_word_num = -1

    for doc_num, words in iteritems(documents):
        if process_log_step and doc_num % process_log_step == 0:
            print('Processed documents:', doc_num)

        cnt = Counter()
        cnt_test = Counter()

        for word_num, number in words:
            max_word_num = max(max_word_num, word_num)
            for _ in range(number):
                if (
                        train_proportion is None
                        or random_gen.random() < train_proportion
                ):
                    cnt[word_num] += 1
                else:
                    cnt_test[word_num] += 1

        if len(cnt) > 0 and (train_proportion is None or len(cnt_test) > 0):
            for w, c in iteritems(cnt):
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)

            for w, c in iteritems(cnt_test):
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)

            not_empty_docs_number += 1

    print('Nonzero train values:', len(data))
    print('Nonzero test values:', len(data_test))

    shape = (not_empty_docs_number, max_word_num + 1)
    if train_proportion is None:
        return scipy.sparse.csr_matrix(
            (data, (row, col)),
            shape=shape
        )
    else:
        return (
            scipy.sparse.csr_matrix(
                (data, (row, col)),
                shape=shape
            ),
            scipy.sparse.csr_matrix(
                (data_test, (row_test, col_test)),
                shape=shape
            )
        )
