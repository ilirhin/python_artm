from artm.datasets import main_cases

import manager


if __name__ == '__main__':
    (
        train_n_dw_matrix,
        test_n_dw_matrix,
        _,
        num_2_token
    ) = main_cases.get_twitter_sentiment140(
        train_proportion=0.8, min_docs_occurrences=3
    )
    manager.perform_experiment(
        train_n_dw_matrix, test_n_dw_matrix, 10, num_2_token
    )
