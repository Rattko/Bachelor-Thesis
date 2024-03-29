#!/usr/bin/env python3

import importlib
import json
import lzma
import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scipy.stats as ss
from mlflow.entities.run import Run
from tqdm import tqdm

from core.utils import get_resampler_name


# Colours for violin plots
YELLOW = '#ECB53A'
RED = '#B52440'


# Names of datasets and their IDs ordered by the level of imbalance in descending order
datasets = {
    '42252': 'Asteroid',
    '4154': 'Credit Card Subset',
    '1597': 'Credit Card',
    '1069': 'PC2',
    '1056': 'MC1',
    '43551': 'Employee Turnover',
    '40900': 'Satellite',
    '1178': 'BNG - Solar Flare',
    '310': 'Mammography',
    'ids': 'CIC-IDS-2017',
    '977': 'Letter',
    '42680': 'Relevant Images',
    '1216': 'Click Prediction V1',
    '1217': 'Click Prediction V2',
    'unsw': 'UNSW-NB15',
    '4135': 'Amazon Employee',
    '131': 'BNG - Sick',
    '1040': 'Sylva Prior',
    '1180': 'BNG - Spect',
    'pdf': 'CIC-Evasive-PDF',
    'ember': 'Ember',
    '3': 'Graph - Embedding',
    '2': 'Graph - Raw'
}

# Names of preprocessing methods in the order of appearance in the thesis
preprocessings = [
    'baseline', 'random_oversampling', 'smote', 'borderline_smote', 'svm_smote',
    'k_means_smote', 'adasyn', 'random_undersampling', 'condensed_nearest_neighbour',
    'edited_nearest_neighbours', 'repeated_edited_nearest_neighbours', 'all_knn', 'near_miss',
    'tomek_links', 'one_sided_selection', 'neighbourhood_cleaning_rule', 'cluster_centroids'
]

# Names of preprocessing methods in the order of appearance in the thesis, prettified
preprocessings_pretty = [
    'Baseline', 'Random Oversampling', 'SMOTE', 'Borderline SMOTE', 'SVM SMOTE',
    'KMeans SMOTE', 'ADASYN', 'Random Undersampling', 'CNN', 'ENN', 'Repeated ENN',
    'All KNN', 'Near Miss', 'Tomek Links', 'One-Sided Selection', 'NCL', 'Cluster Centroids'
]

metrics = [
    'balanced_accuracy', 'precision', 'recall', 'f1_max',
    'matthews_corr_coef', 'pr_auc', 'roc_auc', 'partial_roc_auc'
]

metrics_pretty = [
    'B. Accuracy', 'Precision', 'Recall', 'F1 Max', 'MCC', 'PR AUC', 'ROC AUC', 'P-ROC AUC'
]


def datasets_stats(to_latex: bool = False) -> pd.DataFrame:
    stats = pd.DataFrame(columns=['name', 'size', 'majority_size', 'minority_size', 'imbalance'])

    for dataset_filename in tqdm(datasets.keys(), desc='Generating Datasets Stats', leave=False):
        with lzma.open(f'./../datasets/{dataset_filename}.pickle', 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)

            dataset.imbalance = dataset.majority_size / dataset.minority_size

        stats.loc[len(stats)] = dataset.to_dict()

    stats.sort_values(by='imbalance', ascending=False, inplace=True)

    stats['name'] = (datasets.values())
    stats.columns = ['Name', 'Size', 'Majority Samples', 'Minority Samples', 'Imbalance']

    if to_latex:
        styler = stats.style.format(na_rep='N/A', precision=3).hide()
        styler.to_latex('./tables/datasets.tex', hrules=True, label='table:datasets')

    return stats


def preprocessings_configurations(to_latex: bool = False) -> pd.DataFrame:
    configs = pd.DataFrame(columns=['Method', 'Hyperparameter Configurations'])

    for preprocessing, preprocessing_pretty in zip(preprocessings, preprocessings_pretty):
        preproc_module = importlib.import_module(f'core.preprocessings.{preprocessing}')
        resampler_cls = getattr(preproc_module, get_resampler_name(preprocessing))

        configs.loc[len(configs)] = [preprocessing_pretty, len(list(resampler_cls.hyperparams()))]

    configs.loc[len(configs)] = [r'$\Sigma$', sum(configs['Hyperparameter Configurations'])]

    if to_latex:
        styler = configs.style.format(na_rep='N/A', precision=3).hide()
        styler.to_latex('./tables/configurations.tex', hrules=True, label='table:configs')

    return configs


def plot_preproc_times(runs: list[Run], save: bool = False) -> None:
    times = {key: [] for key in preprocessings[::-1]}

    for run in tqdm(runs, desc='Gathering Preprocessing Times', leave=False):
        preproc = run.data.tags['preprocessing']

        if preproc != 'baseline':
            times[preproc].append(float(run.data.metrics['preprocessing_time']))

    fig = plt.figure(figsize=(8.2, 11.6))

    non_empty_times = [val for val in times.values() if val != []]
    ticks_pretty = [
        preprocessings_pretty[preprocessings.index(name)]
        for name, val in times.items() if val != []
    ]

    parts = plt.violinplot(
        non_empty_times, vert=False, showmeans=True,
        quantiles=[[0.25, 0.5, 0.75]] * len(non_empty_times)
    )

    plt.xscale('log')
    plt.yticks(np.arange(1, len(non_empty_times) + 1), ticks_pretty, rotation=45)

    # Set colours of violin plots
    parts['cmeans'].set_color(RED)
    parts['cquantiles'].set_color(YELLOW)

    # Save as .pdf instead of .eps as the plot needs transparency
    if save:
        plt.savefig('./figures/preprocessing_times.pdf', dpi=800, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


def plot_ranks_distribution(scores: defaultdict[str, pd.DataFrame], save: bool = False) -> None:
    ranks = [frame.rank(ascending=False) for frame in scores.values()]

    for metric in metrics:
        scores = {
            key: np.array([rank.loc[key, metric] for rank in ranks])
            for key in preprocessings[::-1]
        }

        fig = plt.figure(figsize=(8.2, 11.6))

        non_empty_scores = [
            val[~np.isnan(val)] for val in scores.values()
            if len(val) != 0 or np.all(np.isnan(val))
        ]
        ticks_pretty = [
            preprocessings_pretty[preprocessings.index(name)]
            for name, val in scores.items() if len(val) != 0
        ]

        parts = plt.violinplot(
            non_empty_scores, vert=False, showmeans=True,
            quantiles=[[0.25, 0.5, 0.75]] * len(non_empty_scores)
        )

        plt.xlim(0.9, 18.1)
        plt.yticks(np.arange(1, len(non_empty_scores) + 1), ticks_pretty, rotation=45)

        # Set colours of violin plots
        parts['cmeans'].set_color(RED)
        parts['cquantiles'].set_color(YELLOW)

        # Save as .pdf instead of .eps as the plot needs transparency
        if save:
            plt.savefig(f'./figures/{metric}_ranks_distribution.pdf', dpi=800, bbox_inches='tight')
        else:
            plt.show()

        plt.close(fig)


def preprocessing_time(runs: list[Run], to_latex: bool = False) -> pd.DataFrame:
    times = pd.DataFrame(index=preprocessings, columns=list(datasets.keys()))

    for run in tqdm(runs, desc='Gathering Preprocessing Times', leave=False):
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        if 'preprocessing_time' in run.data.metrics.keys():
            times.loc[preproc, dataset] = run.data.metrics['preprocessing_time']
        else:
            times.loc[preproc, dataset] = 0.0

    times.index = preprocessings_pretty
    times.columns = list(datasets.values())

    if to_latex:
        styler = times.style.format(na_rep='N/A', precision=3)
        styler.to_latex('./tables/preprocessing_time.tex', hrules=True, label='table:times')

    return times


def calculate_mean_rank(
    scores: defaultdict[str, pd.DataFrame], to_latex: bool = False
) -> pd.DataFrame:
    ranks = [frame.rank(ascending=False) for frame in scores.values()]
    mean_rank = pd.DataFrame(index=ranks[0].index, columns=ranks[0].columns)

    for row in ranks[0].index:
        for col in ranks[0].columns:
            rank_sum, rank_count = 0, 0

            for frame in ranks:
                value = frame.loc[row, col]

                if not np.isnan(value):
                    rank_sum += value
                    rank_count += 1

            if rank_count == 0:
                mean_rank.loc[row, col] = 'N/A'
            else:
                mean_rank.loc[row, col] = f'{rank_sum / rank_count :.3f} // {rank_count:0>2}'

    mean_rank.index = preprocessings_pretty
    mean_rank.columns = metrics_pretty

    if to_latex:
        with open('./tables/mean_rank.tex', 'w', encoding='utf-8') as table:
            contents = (
                r'\clearpage' '\n'
                r'\begin{table}' '\n'
                r'    \centering' '\n'
                r'    \setlength\tabcolsep{2pt}' '\n'
                r'    \widesplit{' '\n'
                r'        \makebox[\textwidth]{' '\n'
                r'            \begin{tabularx}{\textwidth}{lRRRR}' '\n'
                f'                {mean_rank.style.to_latex(hrules=True)}'
                r'            \end{tabularx}' '\n'
                r'        }' '\n'
                r'    }' '\n'
                r'    \caption{\textbf{Mean Rank Across All Datasets.}}' '\n'
                r'    \label{table:mean_rank}' '\n'
                r'\end{table}'
            )

            table.write(contents)

    return mean_rank


def calculate_f1_max(artifact_uri: str) -> float:
    mlflow.artifacts.download_artifacts(f'{artifact_uri}/data_pr_curve.json', dst_path='.')

    with open('data_pr_curve.json', 'r', encoding='utf-8') as data_file:
        data = json.load(data_file)

    os.remove('data_pr_curve.json')

    data.pop('thresholds')
    scores = pd.DataFrame.from_dict(data)

    scores['f1_max'] = (
        2 * scores['precision'] * scores['recall'] / (scores['precision'] + scores['recall'])
    )

    return max(scores['f1_max'])


def friedman_test(tables: defaultdict[str, pd.DataFrame]) -> pd.DataFrame:
    scores = defaultdict(list)
    results = pd.DataFrame(index=['Statistic', 'P Value'], columns=metrics)

    for frame in tables.values():
        for metric in metrics:
            scores[metric].append(frame.loc[:, metric])

    for metric, metric_scores in scores.items():
        metric_scores = pd.concat(metric_scores, axis=1).T
        result = ss.friedmanchisquare(*[metric_scores[col] for col in metric_scores.columns])

        results.loc['Statistic', metric] = result.statistic
        results.loc['P Value', metric] = result.pvalue

    results.columns = metrics_pretty

    return results


def scores_by_dataset(
    runs: list[Run], ranks_distribution: bool = False, compute_mean_rank: bool = False,
    test: bool = False, to_latex: bool = False
) -> defaultdict[str, pd.DataFrame]:
    tables: defaultdict[str, pd.DataFrame] = defaultdict(
        lambda: pd.DataFrame(index=preprocessings, columns=metrics)
    )

    for run in tqdm(runs, desc='Gathering Scores', leave=False):
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        for metric in metrics:
            if metric == 'f1_max':
                tables[dataset].loc[preproc, metric] = np.nanmax(
                    [tables[dataset].loc[preproc, metric], calculate_f1_max(run.info.artifact_uri)]
                )
                continue

            tables[dataset].loc[preproc, metric] = np.nanmax(
                [tables[dataset].loc[preproc, metric], run.data.metrics[metric]]
            )

    if ranks_distribution:
        plot_ranks_distribution(tables, save=to_latex)

    if compute_mean_rank:
        calculate_mean_rank(tables, to_latex=to_latex)

    if test:
        friedman_test(tables)

    for dataset, frame in tables.items():
        frame.index = preprocessings_pretty
        frame.columns = metrics_pretty

        if to_latex:
            styler = frame.style.format(na_rep='N/A', precision=3) \
                        .highlight_max(props='textbf:--rwrap;')

            with open(f'./tables/{dataset}_metrics.tex', 'w', encoding='utf-8') as table:
                contents = (
                    r'\clearpage' '\n'
                    r'\begin{table}' '\n'
                    r'    \centering' '\n'
                    r'    \setlength\tabcolsep{2pt}' '\n'
                    r'    \widesplit{' '\n'
                    r'        \makebox[\textwidth]{' '\n'
                    r'            \begin{tabularx}{\textwidth}{lRRRR}' '\n'
                    f'                {styler.to_latex(hrules=True)}'
                    r'            \end{tabularx}' '\n'
                    r'        }' '\n'
                    r'    }' '\n'
                    fr'    \caption{{\textbf{{{datasets[dataset]}}}}}' '\n'
                    r'    \label{table:}' '\n'
                    r'\end{table}'
                )

                table.write(contents)

    return tables


def smotes_comparison(runs: list[Run], to_latex: bool = False) -> pd.DataFrame:
    smotes = ['smote', 'borderline_smote', 'svm_smote', 'adasyn']
    metrics = ['pr_auc', 'roc_auc', 'partial_roc_auc']

    tables: defaultdict[str, pd.DataFrame] = defaultdict(
        lambda: pd.DataFrame(index=preprocessings, columns=metrics)
    )

    for run in tqdm(runs, desc='Gathering Scores', leave=False):
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        for metric in metrics:
            tables[dataset].loc[preproc, metric] = np.nanmax(
                [tables[dataset].loc[preproc, metric], run.data.metrics[metric]]
            )

    mean_scores = pd.DataFrame(index=smotes, columns=metrics)
    for smote_variant in smotes:
        for metric in metrics:
            mean_scores.loc[smote_variant, metric] = np.nanmean(
                [scores.loc[smote_variant, metric] for scores in tables.values()]
            )

    multi_index = pd.MultiIndex.from_product([smotes, metrics], names=['Method', 'Metric'])
    relative_diffs = pd.DataFrame(index=multi_index, columns=smotes)
    for i, variant_x in enumerate(smotes):
        for j, variant_y in enumerate(smotes):
            if i < j:
                for metric in metrics:
                    relative_diffs.loc[(variant_x, metric), variant_y] = (
                        mean_scores.loc[variant_x, metric] - mean_scores.loc[variant_y, metric]
                    )

    if to_latex:
        smotes_pretty = {
            'smote': 'SMOTE', 'borderline_smote': 'Borderline SMOTE',
            'svm_smote': 'SVM SMOTE', 'adasyn': 'ADASYN'
        }
        metrics_pretty = {'pr_auc': 'PR AUC', 'roc_auc': 'ROC AUC', 'partial_roc_auc': 'P-ROC AUC'}

        relative_diffs = relative_diffs.rename(
            index={**smotes_pretty, **metrics_pretty}, columns=smotes_pretty
        )

        styler = relative_diffs.style.format(na_rep='-', precision=3) \
                    .highlight_max(props='textbf:--rwrap;')

        with open(f'./tables/smote_relative_diffs.tex', 'w', encoding='utf-8') as table:
            contents = (
                r'\clearpage' '\n'
                r'\begin{table}' '\n'
                r'    \centering' '\n'
                r'    \setlength\tabcolsep{2pt}' '\n'
                r'        \begin{tabularx}{\textwidth}{lRRRR}' '\n'
                f'            {styler.to_latex(hrules=True)}'
                r'        \end{tabularx}' '\n'
                r'    \label{table:}' '\n'
                r'\end{table}'
            )

            table.write(contents)

    return relative_diffs


def main(tracking_uri: str, experiment_name: str) -> None:
    # Set the font in Matplotlib to be the same as in LaTex
    plt.rcParams.update({'font.family': 'Times New Roman'})

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    runs_info = [
        run for run in mlflow.list_run_infos(experiment.experiment_id, max_results=10000)
        if run.status == 'FINISHED'
    ]
    runs = [
        mlflow.get_run(run_info.run_id)
        for run_info in tqdm(runs_info, desc='Gathering Runs', leave=False)
    ]

    plot_preproc_times(runs, save=True)
    scores_by_dataset(
        runs, ranks_distribution=True, compute_mean_rank=True, test=True, to_latex=True
    )


if __name__ == '__main__':
    main('http://127.0.0.1:5000', 'Benchmark')
