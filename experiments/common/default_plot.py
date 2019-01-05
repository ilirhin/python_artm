from __future__ import print_function

import pickle
import re
import sys

from future.utils import itervalues
from future.utils import iterkeys

import numpy as np
import matplotlib.pyplot as plt


def plot_mean(values, iters_count=None):
    samples = len(values)
    if iters_count is None:
        iters_count = len(values[0])
    iter_range = range(1, iters_count + 1)
    val = np.mean(values, axis=0)
    err = 1.96 * np.std(values, axis=0) / np.sqrt(samples)
    plt.plot(iter_range, val[:iters_count], linewidth=2)
    plt.fill_between(
        iter_range, (val - err)[:iters_count], (val + err)[:iters_count],
        alpha=0.5, facecolor='yellow'
    )


def compare(
        values_list, ylabel='', legend=[],
        figsize=(10, 4), title='', iters_count=None
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if iters_count is None:
        iters_count = len(values_list[0][0]) if values_list else 100

    major_ticks = np.arange(0, iters_count + 1, 5)
    ax.set_xticks(major_ticks)
    plt.ylim(0., np.max(np.mean(values_list, axis=1)[:, :iters_count]) * 1.1)
    for values in values_list:
        plot_mean(values, iters_count=iters_count)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.grid()
    plt.title(title)
    plt.show()


def eval_experiment_res(*name_path_pairs, **kwargs):
    iters_count = kwargs.get('iters_count')
    data = dict()
    for name, path in zip(name_path_pairs[::2], name_path_pairs[1::2]):
        try:
            with open(path, 'r') as f:
                data[name] = pickle.load(f)
        except IOError:
            print('Cannot load results from {}'.format(path), file=sys.stderr)

    if all('train_perplexity' in data_dict for data_dict in itervalues(data)):
        use_test_perplexity = all(
            'test_perplexity' in data_dict for data_dict in itervalues(data))
        compare(
            values_list=[
                data_dict['train_perplexity']
                for data_dict in itervalues(data)
            ] + [
                data_dict['test_perplexity']
                for data_dict in itervalues(data)
            ] if use_test_perplexity else [],
            ylabel='Perplexity',
            legend=[
                name + ' train' if use_test_perplexity else ''
                for name in iterkeys(data)
            ] + [
                name + ' test'
                for name in iterkeys(data)
            ] if use_test_perplexity else [],
            iters_count=iters_count
        )

    for metric in data[list(iterkeys(data))[0]]:
        parsed = re.findall('top_\[([\d,]+)\]_pmi', metric)
        if parsed:
            top_sizes = parsed[0].split(',')
            for m_num in [0, 1]:
                for index in range(len(top_sizes)):
                    compare(
                        values_list=[
                            [
                                [
                                    data_dict[metric][sample_num][iter_num][
                                        m_num
                                    ][index]
                                    for iter_num in
                                    range(len(data_dict[metric][sample_num]))
                                ]
                                for sample_num in range(len(data_dict[metric]))

                            ]
                            for data_dict in itervalues(data)
                        ],
                        ylabel='Top {} {}'.format(
                            top_sizes[index],
                            'PPMI' if m_num else 'PMI'
                        ),
                        legend=list(iterkeys(data)),
                        iters_count=iters_count
                    )

        values_list = [data_dict[metric] for data_dict in itervalues(data)]
        legend = list(iterkeys(data))

        parsed = re.findall('top_(\d+)_avg_jaccard', metric)
        if parsed:
            top_size = parsed[0]
            compare(
                values_list=values_list,
                ylabel='Top {} Average jaccard'.format(top_size),
                legend=legend,
                iters_count=iters_count
            )

        if metric in {
            'kernel_avg_size', 'kernel_avg_jaccard',
            'phi_sparsity', 'theta_sparsity',
            'topic_correlation'
        }:
            compare(
                values_list=values_list,
                ylabel=metric.replace('_', ' ').title(),
                legend=legend,
                iters_count=iters_count
            )
