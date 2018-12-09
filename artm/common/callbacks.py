from future.utils import iteritems

from collections import defaultdict
import os
import pickle

import numpy as np

from artm.calculations import metrics
from artm.common.timers import SimpleTimer


def save_results(result_obj, output_path):
    """
    :param result_obj: the object to be saved
    :param output_path: path where to save result_obj
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    with open(output_path, 'w') as output_file:
        pickle.dump(result_obj, output_file)


class Basic(object):
    def start_launch(self):
        """
        :return:

        starts series of the calls
        """
        pass

    def finish_launch(self):
        """

        :return:

        finishes series of the calls
        """
        pass

    def __call__(self, it, phi, theta):
        """
        :param it: the number of the iteration
        :param phi: topics-words matrix, shape T x W, stochastic over W
        :param theta: docs-topics matrix, shape D x T, stochastic over T
        :return:
        """
        raise NotImplementedError()

    def save_results(self, output_path):
        """
        :param output_path: path where to save
        :return:

        saves results of the calls
        """
        pass


class Callback(Basic):
    def __init__(self, metrics):
        self.metrics = metrics
        self.result = defaultdict(list)
        self.launch_result = None

    def start_launch(self):
        self.launch_result = defaultdict(list)

    def finish_launch(self):
        for name, values in iteritems(self.launch_result):
            self.result[name].append(values)
        self.launch_result = None

    def __call__(self, it, phi, theta):
        for name, metric in iteritems(self.metrics):
            self.launch_result[name].append(metric(it, phi, theta))

    def save_results(self, output_path):
        save_results(self.result, output_path)


class TimedCallback(Callback):
    def __call__(self, it, phi, theta):
        for name, metric in iteritems(self.metrics):
            with SimpleTimer(name):
                self.launch_result[name].append(metric(it, phi, theta))


class Builder(object):
    """
    Builder of callbacks. It makes the creation of a callback more simple
    """
    def __init__(self, measure_time=False):
        self.metrics = dict()
        self.measure_time = measure_time

    def top_avg_jaccard(self, top_size):
        self.metrics[
            'top_{}_avg_jaccard'.format(top_size)
        ] = lambda it, phi, theta: metrics.calc_avg_top_words_jaccards(
            phi, top_size
        )
        return self

    def perplexity(self, name, n_dw_matrix):
        calc = metrics.calc_perplexity_function(n_dw_matrix)
        self.metrics[name] = lambda it, phi, theta: calc(
            phi, theta
        )
        return self

    def top_pmi(
        self, doc_occurences, doc_cooccurences,
        documents_number, top_sizes, cooccurences_smooth=1.
    ):
        calc = metrics.create_pmi_top_function(
            doc_occurences, doc_cooccurences, documents_number,
            top_sizes, cooccurences_smooth
        )
        self.metrics[
            'top_[{}]_pmi'.format(','.join(map(str, sorted(top_sizes))))
        ] = lambda it, phi, theta: calc(phi)
        return self

    def kernel_avg_size(self):
        self.metrics[
            'kernel_avg_size'
        ] = lambda it, phi, theta: np.mean(metrics.calc_kernels_sizes(phi))
        return self

    def kernel_avg_jaccard(self):
        self.metrics[
            'kernel_avg_jaccard'
        ] = lambda it, phi, theta: metrics.calc_avg_pairwise_kernels_jaccards(
            phi
        )
        return self

    def sparsity(self):
        self.metrics[
            'phi_sparsity'
        ] = lambda it, phi, theta: 1. * np.sum(phi == 0) / np.sum(phi >= 0)
        return self

    def theta_sparsity(self):
        self.metrics[
            'theta_sparsity'
        ] = lambda it, phi, theta: 1. * np.sum(theta == 0) / np.sum(theta >= 0)
        return self

    def topic_correlation(self):
        self.metrics[
            'topic_correlation'
        ] = lambda it, phi, theta: metrics.calc_topics_correlation(phi)
        return self

    def uniqueness_measure(self):
        self.metrics[
            'uniqueness_measure'
        ] = lambda it, phi, theta: metrics.calc_phi_uniqueness_measures(phi)
        return self

    def phi(self):
        self.metrics[
            'phi'
        ] = lambda it, phi, theta: phi
        return self

    def theta(self):
        self.metrics[
            'theta'
        ] = lambda it, phi, theta: theta
        return self

    def build(self):
        if self.measure_time:
            return TimedCallback(self.metrics)
        else:
            return Callback(self.metrics)
