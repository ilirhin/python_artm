import pickle

import numpy as np

from pyartm import common
from pyartm.common import experiments
from pyartm_datasets import main_cases
from pyartm import regularizers
from pyartm.optimizations import default


if __name__ == '__main__':
    # create dataset matrices, you can write custom source here
    train_n_dw_matrix, test_n_dw_matrix = main_cases.get_20newsgroups([
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ], train_proportion=0.8)[:2]
    # create list of regularizers (one for each iteration)
    # its a callable object with interface
    # __call__(self, phi, theta, n_tw, n_dt)
    # where phi and theta or new value at the start of the EM iteration
    # and n_tw, n_dt are counters calculated on the iteration
    iters_count = 100
    regularization_list = [regularizers.Additive(0.1, 0.1)] * iters_count
    # here we create optimizer which inherits
    # pyartm.optimizations.base.Optimizer and must implement the method
    #    def _run(self, n_dw_matrix, docptr, wordptr, phi_matrix, theta_matrix):
    #    """
    #    :param n_dw_matrix: documents-words matrix D x W
    #    :param docptr: docptr for n_dw_matrix
    #       (for all occurrences we store its document number)
    #    :param wordptr: wordptr for n_dw_matrix
    #       (for all occurrences we stote its word number)
    #    :param phi_matrix: matrix T x W, can be modified
    #    :param theta_matrix: D x T, can be modified
    #    :return: phi_matrix, theta_matrix, n_tw, n_dt after optimization
    optimizer = default.Optimizer(regularization_list)
    # at the end of each iteration the iteration_callback will be called
    # experiments.default_callback calculates different statistics and saves
    # them to list, as a result you will have the values of
    # the specified statistics at the end of each iteration
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix,
        top_pmi_sizes=[5, 10, 20, 30],
        top_avg_jaccard_sizes=[10, 50, 100, 200]
    )

    # we create the initial values for Phi and Theta
    D, W = train_n_dw_matrix.shape
    T = 10
    random_gen = np.random.RandomState(42)
    phi_matrix = common.get_prob_matrix_by_counters(
        random_gen.uniform(size=(T, W)).astype(np.float64)
    )
    theta_matrix = common.get_prob_matrix_by_counters(
        np.ones(shape=(D, T)).astype(np.float64)
    )
    # we indicate the end start of the launch
    # (it's useful if you have many launches)
    optimizer.iteration_callback.start_launch()
    # we launch optimization via the optimizer object
    result = optimizer.run(train_n_dw_matrix, phi_matrix, theta_matrix)
    # we indicate the end of the launch
    optimizer.iteration_callback.finish_launch()
    # save the results stored in the iteration_callback (default is via pickle)
    optimizer.iteration_callback.save_results(
        '20newsgroups_example_callback_results.pkl'
    )
    # save result variable as pickle
    with open('20newsgroups_example_result.pkl', 'w') as output:
        pickle.dump(result, output)
