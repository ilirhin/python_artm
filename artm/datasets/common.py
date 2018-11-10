import random
from collections import Counter

import scipy.sparse


def create_sparse_matrices(documents, test_proportion=None, process_log_step=None, random_seed=42):
    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    random_gen = random.Random(random_seed)
    max_word_num = -1

    for doc_num, words in documents.iteritems():
        if process_log_step and doc_num % process_log_step == 0:
            print 'Processed documents:', doc_num

        cnt = Counter()
        cnt_test = Counter()

        for word_num, number in words:
            max_word_num = max(max_word_num, word_num)
            for _ in xrange(number):
                if (
                        test_proportion is None
                        or random_gen.random() < test_proportion
                ):
                    cnt[word_num] += 1
                else:
                    cnt_test[word_num] += 1

        if len(cnt) > 0 and (test_proportion is None or len(cnt_test) > 0):
            for w, c in cnt.iteritems():
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)

            for w, c in cnt_test.iteritems():
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)

            not_empty_docs_number += 1

    shape = (not_empty_docs_number, max_word_num + 1)
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
