from pyartm.common import experiments


def perform_alpha_dependency_experiment((
    train_n_dw_matrix, optimizer, T, samples, output_path
)):
    callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        uniqueness_measures=True
    )
    callback.start_launch()
    for seed in range(samples):
        print(seed)
        callback(0, *experiments.default_sample(
            train_n_dw_matrix, T, seed, optimizer
        ))
    callback.finish_launch()
    callback.save_results(output_path)


def perform_iteration_dependency_experiment((
    train_n_dw_matrix, test_n_dw_matrix, optimizer,
    T, samples, output_path
)):
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix,
        uniqueness_measures=True
    )
    for seed in range(samples):
        print(seed)
        experiments.default_sample(train_n_dw_matrix, T, seed, optimizer)
    optimizer.iteration_callback.save_results(output_path)
