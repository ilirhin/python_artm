import pickle
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from artm import common
from artm.common import experiments
from artm.common import callbacks
from artm.optimizations import thetaless
from artm.optimizations import default
from artm.calculations import metrics


def perform_experiment((
   train_n_dw_matrix, test_n_dw_matrix, optimizer,
   T, samples, output_path
)):
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix,
        top_pmi_sizes=[5, 10, 20, 30],
        top_avg_jacard_sizes=[10, 50, 100, 200]
    )
    for seed in range(samples):
        print(seed)
        experiments.default_sample(train_n_dw_matrix, T, seed, optimizer)
    optimizer.iteration_callback.save_results(output_path)


def perform_doc_experiment((
    n_dw_matrix_doc_train, doc_targets_doc_train,
    n_dw_matrix_doc_test, doc_targets_doc_test,
    optimizer, T, samples, output_path
)):
    D, _ = n_dw_matrix_doc_test.shape
    svm_train_score = metrics.create_svm_score_function(n_dw_matrix_doc_train)
    opt_plsa_not_const_phi = default.Optimizer(
        regularization_list=optimizer.regularization_list[:10],
        const_phi=False
    )
    opt_plsa_const_phi = default.Optimizer(
        regularization_list=optimizer.regularization_list[:10],
        const_phi=True
    )
    opt_artm_thetaless = thetaless.Optimizer(
        regularization_list=optimizer.regularization_list[:10]
    )

    res_plsa_not_const_phi = []
    res_plsa_const_phi = []
    res_artm_thetaless = []
    cv_fold_scores = []
    cv_test_scores = []

    for seed in range(samples):
        print(seed)
        phi, theta = experiments.default_sample(
            n_dw_matrix_doc_train, T, seed, optimizer
        )

        (
            best_C, best_gamma,
            cv_fold_score, cv_test_score
        ) = svm_train_score(theta)
        cv_fold_scores.append(cv_fold_score)
        cv_test_scores.append(cv_test_score)

        print('Fold score: {}\tTest score: {}'.format(
            cv_fold_score, cv_test_score
        ))
        algo = SVC(C=best_C, gamma=best_gamma).fit(
            theta, doc_targets_doc_train
        )
        init_theta = common.get_prob_matrix_by_counters(
            np.ones(shape=(D, T), dtype=np.float64)
        )

        plsa_not_const_phi = []
        plsa_const_phi = []
        artm_thetaless = []

        opt_plsa_not_const_phi.iteration_callback = (
            lambda it, phi, theta:
            plsa_not_const_phi.append(
                accuracy_score(algo.predict(theta), doc_targets_doc_test)
            )
        )
        opt_plsa_const_phi.iteration_callback = (
            lambda it, phi, theta:
            plsa_const_phi.append(
                accuracy_score(algo.predict(theta), doc_targets_doc_test)
            )
        )
        opt_artm_thetaless.iteration_callback = (
            lambda it, phi, theta:
            artm_thetaless.append(
                accuracy_score(algo.predict(theta), doc_targets_doc_test)
            )
        )

        for opt in [
            opt_plsa_not_const_phi, opt_plsa_const_phi, opt_artm_thetaless
        ]:
            opt.run(n_dw_matrix_doc_test, phi, init_theta)

        res_plsa_not_const_phi.append(plsa_not_const_phi)
        res_plsa_const_phi.append(plsa_const_phi)
        res_artm_thetaless.append(artm_thetaless)

    callbacks.save_results({
        'res_plsa_not_const_phi': res_plsa_not_const_phi,
        'res_plsa_const_phi': res_plsa_const_phi,
        'res_artm_thetaless': res_artm_thetaless,
        'cv_fold_scores': cv_fold_scores,
        'cv_test_scores': cv_test_scores
    }, output_path)
