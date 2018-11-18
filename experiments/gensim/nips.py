from artm.datasets import main_cases

import manager


if __name__ == '__main__':
    (
        train_n_dw_matrix,
        test_n_dw_matrix,
        _,
        num_2_token
    ) = main_cases.get_nips(train_proportion=0.8)
    manager.perform_experiment(
        train_n_dw_matrix, test_n_dw_matrix, 10, num_2_token
    )
