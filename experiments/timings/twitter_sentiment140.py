from pyartm import regularizers
from pyartm.datasets import main_cases
from pyartm.optimizations import timed_default

import manager


if __name__ == '__main__':
    manager.perform_experiment(
        main_cases.get_twitter_sentiment140(min_docs_occurrences=3)[0],
        timed_default.Optimizer(
            regularization_list=[regularizers.Additive(0., 0.)] * 100,
            return_counters=True
        ), 10, 100
    )
