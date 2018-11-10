from __future__ import print_function

import random
from collections import Counter
from collections import defaultdict

import scipy
import scipy.sparse


def prepare(
        dataset_path,
        test_proportion=None,
        process_log_step=1000,
        early_stop=None
):
    token_2_num = {}
    documents = defaultdict(list)

    with open(dataset_path, 'r') as dataset_file:
        for index, line in enumerate(dataset_file):
            if index % process_log_step == 0:
                print('Read file lines:', index)
            if early_stop and index >= early_stop:
                break

            if index > 0:
                tokens = line.strip().split(',')
                token_2_num[tokens[0][1:-1]] = index - 1
                for doc_num, val in enumerate(tokens[1:]):
                    value = int(val)
                    if value > 0:
                        documents[doc_num].append((index - 1, value))

    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }

    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    random_gen = random.Random(42)
    not_empty_docs_number = 0

    for doc_num, words in documents.iteritems():
        if doc_num % process_log_step == 0:
            print('Processed documents:', doc_num)

        cnt = Counter()
        cnt_test = Counter()

        for word_num, number in words:
            for _ in xrange(number):
                if (
                        test_proportion is None
                        or random_gen.random() >= test_proportion
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

    print('Nonzero values:', len(data))
    shape = (not_empty_docs_number, len(token_2_num))

    if test_proportion is None:
        return (
            scipy.sparse.csr_matrix((data, (row, col))),
            token_2_num,
            num_2_token
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
            ),
            token_2_num,
            num_2_token
        )
