import numpy as np
from gensim.models.ldamodel import LdaModel

from pyartm import regularizers
from pyartm import common
from pyartm.common import experiments
from pyartm.optimizations import default

import common as exp_common


def perform_experiment(train_n_dw_matrix, test_n_dw_matrix, T, num_2_token):
    train_corpus = [zip(row.indices, row.data) for row in train_n_dw_matrix]

    for seed in [42, 7, 777, 12]:
        model = LdaModel(
            train_corpus,
            alpha='auto', id2word=num_2_token,
            num_topics=T, iterations=500, random_state=seed
        )
        gensim_phi = exp_common.get_phi(model)
        gensim_theta = exp_common.get_theta(train_corpus, model)
        print('gensim perplexity')
        print(np.exp(-model.log_perplexity(train_corpus)))

        D, W = train_n_dw_matrix.shape
        random_gen = np.random.RandomState(seed)
        phi = common.get_prob_matrix_by_counters(
            random_gen.uniform(size=(T, W)).astype(np.float64)
        )
        theta = common.get_prob_matrix_by_counters(
            np.ones(shape=(D, T)).astype(np.float64)
        )
        phi, theta = default.Optimizer(
            [regularizers.Additive(0.1, 0.)] * 100,
            verbose=False
        ).run(
            train_n_dw_matrix, phi, theta
        )

        callback = experiments.default_callback(
            train_n_dw_matrix=train_n_dw_matrix,
            test_n_dw_matrix=test_n_dw_matrix,
            top_pmi_sizes=[5, 10, 20, 30],
            top_avg_jaccard_sizes=[10, 50, 100, 200],
            measure_time=True
        )
        callback.start_launch()
        callback(0, phi, theta)
        callback(1, gensim_phi, gensim_theta)

        print('artm')
        for name, values in callback.launch_result.items():
            print('\t{}: {}'.format(name, values[0]))

        print('gensim')
        for name, values in callback.launch_result.items():
            print('\t{}: {}'.format(name, values[1]))
