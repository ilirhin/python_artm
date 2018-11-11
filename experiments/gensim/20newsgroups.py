from artm.datasets import main_cases

import manager


if __name__ == '__main__':
    (
        train_n_dw_matrix,
        test_n_dw_matrix,
        _,
        num_2_token,
        _
    ) = main_cases.get_20newsgroups([
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball',
        'rec.sport.hockey',
        'sci.crypt',
        'sci.electronics',
        'sci.med',
        'sci.space'
    ], train_test_split=0.8)
    manager.perform_experiment(
        train_n_dw_matrix, test_n_dw_matrix, 10, num_2_token
    )
