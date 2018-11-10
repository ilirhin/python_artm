# coding: utf-8

from __future__ import print_function

import csv
import scipy
import scipy.sparse
import gensim
import random
from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords


def prepare(
        dataset_path,
        test_proportion=None,
        process_log_step=10000,
        early_stop=None,
        min_docs_occurrences=3
):
    english_stopwords = set(stopwords.words('english'))
    documents = defaultdict(list)
    docs_occurrences = Counter()

    with open(dataset_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for index, line in enumerate(reader):
            if index % process_log_step == 0:
                print('Preprocessed: ', index)
            if early_stop and index >= early_stop:
                break

            doc_target, _, _, _, _, text = line
            tokens = list(gensim.utils.lemmatize(text))
            for token in tokens:
                if token not in english_stopwords:
                    documents[index].append(token)
            for token in set(tokens):
                if token not in english_stopwords:
                    docs_occurrences[token] += 1

    token_2_num = dict()

    for doc_num in documents:
        documents[doc_num] = Counter([
            word
            for word in documents[doc_num]
            if docs_occurrences[word] >= min_docs_occurrences]
        ).items()

        for (word, count) in documents[doc_num]:
            if word not in token_2_num:
                token_2_num[word] = len(token_2_num)
        documents[doc_num] = [(token_2_num[w], c) for (w, c) in documents[doc_num]]

    num_2_token = {
        v: k
        for k, v in token_2_num.iteritems()
    }

    row, col, data = [], [], []
    row_test, col_test, data_test = [], [], []
    not_empty_docs_number = 0
    random_gen = random.Random(42)

    for doc_num, words in documents.iteritems():
        if doc_num % process_log_step == 0:
            print('Processed documents:', doc_num)

        cnt = Counter()
        cnt_test = Counter()

        for word_num, number in words:
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
