from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import default, thetaless
from pyartm.common import experiments
from pyartm.calculations import metrics


def print_matrix(arr):
    for row in arr:
        line = list(map(str, row))
        print(' '.join(line))


if __name__ == '__main__':
    _n_dw_matrix, _, num_2_token, doc_targets = main_cases.get_20newsgroups([
        'rec.sport.hockey',
        'talk.politics.guns',
    ])
    topic_0_indices, topic_1_indices = [],  []
    for index, target in enumerate(doc_targets):
        if target == 0:
            topic_0_indices.append(index)
        elif target == 1:
            topic_1_indices.append(index)

    thetaless_rels = []
    lda_rels = []
    for balance in range(10, 201, 10):
        print(balance)
        n_dw_matrix = _n_dw_matrix[topic_0_indices + topic_1_indices * balance, :]
        regularization_list = [regularizers.Additive(-0.1, 0.)] * 100
        lda_phi, lda_theta = experiments.default_sample(
            n_dw_matrix,
            T=2,
            seed=42,
            optimizer=default.Optimizer(regularization_list, verbose=False)
        )
        thetaless_phi, thetaless_theta = experiments.default_sample(
            n_dw_matrix,
            T=2,
            seed=42,
            optimizer=thetaless.Optimizer(regularization_list, verbose=False)
        )
        # print(np.argmax(thetaless_theta[:len(topic_0_indices), :2], axis=1).mean())
        # print(np.argmax(thetaless_theta[len(topic_0_indices):, :2], axis=1).mean())
        # print('!')
        # for topic_set in metrics.get_top_words(thetaless_phi, 10):
        #     print('\n\t'.join(map(num_2_token.get, topic_set)))
        #     print()
        # for topic_set in metrics.get_top_words(thetaless_phi, 5):
        #     print('\n\t'.join(map(num_2_token.get, topic_set)))
        #     print()
        print('lda')
        # #print(np.sum(lda_theta[:, 1]) / np.sum(lda_theta[:, 0]))
        # print(np.argmax(lda_theta, axis=1).mean())
        print(metrics.calc_avg_top_words_jaccards(lda_phi, 20))
        print('thetaless')
        # #print(np.sum(thetaless_theta[:, 1]) / np.sum(thetaless_theta[:, 0]))
        # print(np.argmax(thetaless_theta, axis=1).mean())
        print(metrics.calc_avg_top_words_jaccards(thetaless_phi, 20))
        print()
