import pickle
import re

import numpy as np
import matplotlib.pyplot as plt


def plot_mean(values):
    samples, iters = len(values), len(values[0])
    iter_range = range(1, iters + 1)
    val = np.mean(values, axis=0)
    err = 1.96 * np.std(values, axis=0) / np.sqrt(samples)
    plt.plot(iter_range, val)
    plt.fill_between(
        iter_range, val - err, val + err, alpha=0.5, facecolor='yellow'
    )


def compare(values_list, ylabel='', legend=[], figsize=(10, 4), title=''):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    major_ticks = np.arange(
        0, len(values_list[0]) + 1 if values_list else 101, 5
    )
    ax.set_xticks(major_ticks)
    plt.ylim(0., np.max(values_list) * 1.01)
    for values in values_list:
        plot_mean(values)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.grid()
    plt.title(title)
    plt.show()


def eval_experiment_res(
    fst_name, fst_path,
    snd_name, snd_path,
):
    with open(fst_path, 'r') as f:
        fst_dict = pickle.load(f)
    with open(snd_path, 'r') as f:
        snd_dict = pickle.load(f)

    if 'train_perplexity' in fst_dict:
        use_test_perplexity = 'test_perplexity' in fst_dict
        compare(
            values_list=[
                fst_dict['train_perplexity'], snd_dict['train_perplexity']
            ] + [
                fst_dict['test_perplexity'], snd_dict['test_perplexity']
            ] if use_test_perplexity else [],
            ylabel='Perplexity',
            legend=[
                fst_name + ' train' if use_test_perplexity else '',
                snd_name + ' train' if use_test_perplexity else ''
            ] + [
                fst_name + ' test', snd_name + ' test'
            ] if use_test_perplexity else []
        )

    for metric in fst_dict:
        parsed = re.findall('top_\[([\d,]+)\]_pmi', metric)
        if parsed:
            top_sizes = parsed[0].split(',')
            for m_num in [0, 1]:
                for index in range(len(top_sizes)):
                    compare(
                        values_list=[
                            [
                                [
                                    fst_dict[metric][sample_num][iter_num][
                                        m_num
                                    ][index]
                                    for iter_num in
                                    range(len(fst_dict[metric][sample_num]))
                                ]
                                for sample_num in range(len(fst_dict[metric]))

                            ],
                            [
                                [
                                    snd_dict[metric][sample_num][iter_num][
                                        m_num
                                    ][index]
                                    for iter_num in
                                    range(len(snd_dict[metric][sample_num]))
                                ]
                                for sample_num in range(len(snd_dict[metric]))
                            ]
                        ],
                        ylabel='Top {} {}'.format(
                            top_sizes[index],
                            'PPMI' if m_num else 'PMI'
                        ),
                        legend=[fst_name, snd_name]
                    )

        parsed = re.findall('top_(\d+)_avg_jaccard', metric)
        if parsed:
            top_size = parsed[0]
            compare(
                values_list=[fst_dict[metric], snd_dict[metric]],
                ylabel='Top {} Average jaccard'.format(top_size),
                legend=[fst_name, snd_name]
            )

        if metric in {
            'kernel_avg_size', 'kernel_avg_jaccard',
            'phi_sparsity', 'theta_sparsity',
            'topic_correlation'
        }:
            compare(
                values_list=[fst_dict[metric], snd_dict[metric]],
                ylabel=metric.replace('_', ' ').title(),
                legend=[fst_name, snd_name]
            )