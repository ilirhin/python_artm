from __future__ import print_function

from builtins import zip
import random
from collections import Counter

import scipy.sparse
import gensim
from nltk.corpus import stopwords


def prepare(
        dataset,
        train_proportion=None,
        min_occurrences=3,
        token_2_num=None,
        process_log_step=500
):
    english_stopwords = set(stopwords.words('english'))
    is_token_2_num_provided = token_2_num is not None

    # remove stopwords
    if not is_token_2_num_provided:
        token_2_num = dict()
        occurrences = Counter()
        for i, doc in enumerate(dataset.data):
            tokens = gensim.utils.lemmatize(doc)
            for token in set(tokens):
                occurrences[token] += 1
            if i % process_log_step == 0:
                print('Preprocessed: ', i, 'documents from', len(dataset.data))

    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    doc_targets = []
    random_gen = random.Random(42)

    for i, (doc, target) in enumerate(
        zip(dataset.data, dataset.target)
    ):
        tokens = gensim.utils.lemmatize(doc)
        cnt = Counter()
        cnt_test = Counter()
        for token in tokens:
            word = token.split('/')[0]
            if (
                    not is_token_2_num_provided
                    and word not in english_stopwords
                    and min_occurrences <= occurrences[token]
                    and token not in token_2_num
            ):
                token_2_num[token] = len(token_2_num)
            if token in token_2_num:
                if (
                        train_proportion is None
                        or random_gen.random() < train_proportion
                ):
                    cnt[token_2_num[token]] += 1
                else:
                    cnt_test[token_2_num[token]] += 1

        if len(cnt) > 0 and (train_proportion is None or len(cnt_test) > 0):
            for w, c in cnt.iteritems():
                row.append(not_empty_docs_number)
                col.append(w)
                data.append(c)

            for w, c in cnt_test.iteritems():
                row_test.append(not_empty_docs_number)
                col_test.append(w)
                data_test.append(c)

            not_empty_docs_number += 1
            doc_targets.append(target)

        if i % process_log_step == 0:
            print('Processed: ', i, 'documents from', len(dataset.data))

    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }

    shape = (not_empty_docs_number, len(token_2_num))
    if train_proportion is None:
        return (
            scipy.sparse.csr_matrix((data, (row, col)), shape=shape),
            token_2_num,
            num_2_token,
            doc_targets
        )
    else:
        return (
            scipy.sparse.csr_matrix((data, (row, col)), shape=shape),
            scipy.sparse.csr_matrix(
                (data_test, (row_test, col_test)), shape=shape
            ),
            token_2_num,
            num_2_token,
            doc_targets
        )
