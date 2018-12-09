from __future__ import print_function
from future.utils import iteritems

from collections import Counter
from collections import defaultdict
import csv

import gensim
from nltk.corpus import stopwords

from .common import create_sparse_matrices


def prepare(
        dataset_path,
        train_proportion=None,
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
        documents[doc_num] = list(Counter([
            word
            for word in documents[doc_num]
            if docs_occurrences[word] >= min_docs_occurrences]
        ).items())

        for (word, count) in documents[doc_num]:
            if word not in token_2_num:
                token_2_num[word] = len(token_2_num)
        documents[doc_num] = [
            (token_2_num[w], c)
            for (w, c) in documents[doc_num]
        ]

    num_2_token = {
        v: k
        for k, v in iteritems(token_2_num)
    }

    matrices = create_sparse_matrices(
        documents,
        train_proportion=train_proportion,
        process_log_step=process_log_step,
        random_seed=42
    )

    return tuple(list(matrices) + [token_2_num, num_2_token])
