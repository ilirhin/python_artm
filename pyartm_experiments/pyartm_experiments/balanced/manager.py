from pyartm.common import experiments


def perform_experiment((
   train_n_dw_matrix, test_n_dw_matrix, optimizer,
   T, samples, output_path
)):
    optimizer.iteration_callback = experiments.default_callback(
        train_n_dw_matrix=train_n_dw_matrix,
        test_n_dw_matrix=test_n_dw_matrix,
        top_pmi_sizes=[5, 10, 20, 30],
        top_avg_jaccard_sizes=[10, 50, 100, 200],
        measure_time=True
    )
    for seed in range(samples):
        print(seed)
        experiments.default_sample(train_n_dw_matrix, T, seed, optimizer)
    optimizer.iteration_callback.save_results(output_path)
