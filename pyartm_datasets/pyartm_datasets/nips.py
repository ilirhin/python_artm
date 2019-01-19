from __future__ import print_function
from future.utils import iteritems

from collections import defaultdict

from .common import create_sparse_matrices


def prepare(
        dataset_path,
        train_proportion=None,
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
        for k, v in iteritems(token_2_num)
    }

    matrices = create_sparse_matrices(
        documents,
        train_proportion=train_proportion,
        process_log_step=process_log_step,
        random_seed=42
    )

    return tuple(list(matrices) + [token_2_num, num_2_token])
