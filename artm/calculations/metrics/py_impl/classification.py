# coding: utf-8

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def create_svm_score_function(
    targets, verbose=True,
    test_size=0.3, random_state=42,
    C_range=None, gamma_range=None
):
    if C_range is None:
        C_range = [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    if gamma_range is None:
        gamma_range = [1e-3, 1e-2, 1e-1, 1, 1e1]

    def func(matrix):
        best_C, best_gamma, best_val = None, None, 0.
        best_cv_algo_score_on_test = 0.
        X_train, X_test, y_train, y_test = train_test_split(
            matrix, targets,
            test_size=test_size,
            stratify=targets,
            random_state=random_state
        )
        for C in C_range:
            for gamma in gamma_range:
                val = cross_val_score(
                    SVC(C=C, gamma=gamma),
                    X_train, y_train,
                    scoring='accuracy',
                    cv=4
                ).mean()
                algo = SVC(C=C, gamma=gamma).fit(X_train, y_train)
                test_score = accuracy_score(y_test, algo.predict(X_test))
                if verbose:
                    log_msg = 'SVM(C={}, gamma={}) cv-score: {}  test-score: {}'
                    print log_msg.format(
                        C,
                        gamma,
                        round(val, 3),
                        round(test_score, 3)
                    )
                if val > best_val:
                    best_val = val
                    best_C = C
                    best_gamma = gamma
                    best_cv_algo_score_on_test = test_score
        if verbose:
            log_msg = (
                    '\n\n\nBest cv params: C={}, gamma={}'
                    + '\nCV score: {}\nTest score:{}'
            )
            print log_msg.format(
                best_C,
                best_gamma,
                round(best_val, 3),
                round(best_cv_algo_score_on_test, 3)
            )
        return best_C, best_gamma, best_val, best_cv_algo_score_on_test

    return func
