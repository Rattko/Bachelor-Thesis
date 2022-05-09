#!/usr/bin/env python3

import importlib
import json
import lzma
import os
import pickle
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.run_info import RunInfo
from tqdm import tqdm

from core.utils import get_resampler_name


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
    '977': 'Letter',
    '42680': 'Relevant Images',
    '1216': 'Click Prediction V1',
    '1217': 'Click Prediction V2',
    '4135': 'Amazon Employee',
    '131': 'BNG - Sick',
    '1040': 'Sylva Prior',
    '1180': 'BNG - Spect'
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
    'balanced_accuracy', 'precision', 'recall', 'f1_max', 'pr_auc',
    'matthews_corr_coef', 'roc_auc', 'partial_roc_auc'
]

metrics_pretty = [
    'B. Accuracy', 'Precision', 'Recall', 'F1 Max', 'PR AUC', 'MCC', 'ROC AUC', 'P-ROC AUC'
]


def datasets_stats(to_latex: bool = False) -> pd.DataFrame:
    stats = pd.DataFrame(columns=['name', 'size', 'majority_size', 'minority_size', 'imbalance'])

    for dataset_filename in tqdm(datasets.keys(), desc='Generating Datasets Stats', leave=False):
        with lzma.open(f'./datasets/{dataset_filename}.pickle', 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)

        stats.loc[len(stats)] = dataset.to_dict()

    stats.sort_values(by='imbalance', ascending=False, inplace=True)

    stats['name'] = (datasets.values())
    stats.columns = ['Name', 'Size', 'Majority Samples', 'Minority Samples', 'Imbalance']

    if to_latex:
        styler = stats.style.format(na_rep='N/A', precision=3).hide()
        styler.to_latex('./thesis/tables/datasets.tex', hrules=True, label='table:datasets')

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
        styler.to_latex('./thesis/tables/configurations.tex', hrules=True, label='table:configs')

    return configs


def preprocessing_time(runs_info: list[RunInfo], to_latex: bool = False) -> pd.DataFrame:
    times = pd.DataFrame(index=preprocessings, columns=list(datasets.keys()))

    for run_info in tqdm(runs_info, desc='Gathering Preprocessing Times', leave=False):
        run = mlflow.get_run(run_info.run_id)
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
        styler.to_latex('./thesis/tables/preprocessing_time.tex', hrules=True, label='table:times')

    return times


def calculate_f1_max(artifact_uri: str) -> float:
    mlflow.artifacts.download_artifacts(f'{artifact_uri}/data_pr_curve.json', dst_path='.')

    with open('data_pr_curve.json', 'r') as data_file:
        data = json.load(data_file)

    os.remove('data_pr_curve.json')

    data.pop('thresholds')
    scores = pd.DataFrame.from_dict(data)

    scores['f1_max'] = (
        2 * scores['precision'] * scores['recall'] / (scores['precision'] + scores['recall'])
    )

    return max(scores['f1_max'])


def ranks_by_dataset(
    scores: defaultdict[str, pd.DataFrame], to_latex: bool = False
) -> defaultdict[str, pd.DataFrame]:
    ranks: defaultdict[str, pd.DataFrame] = defaultdict()

    for dataset, frame in scores.items():
        ranks[dataset] = frame.rank(ascending=False)
        ranks[dataset].index = preprocessings_pretty
        ranks[dataset].columns = metrics_pretty

        if to_latex:
            styler = ranks[dataset].style.format(na_rep='N/A', precision=1) \
                        .highlight_min(props='textbf:--rwrap;')
            styler.to_latex(
                f'./thesis/tables/{dataset}_ranks.tex', hrules=True, label='table:ranks'
            )

    return ranks


def scores_by_dataset(
    runs_info: list[RunInfo], ranks: bool = False, to_latex: bool = False
) -> defaultdict[str, pd.DataFrame]:
    tables: defaultdict[str, pd.DataFrame] = defaultdict(
        lambda: pd.DataFrame(index=preprocessings, columns=metrics)
    )

    for run_info in tqdm(runs_info, desc='Gathering Scores', leave=False):
        run = mlflow.get_run(run_info.run_id)
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        for metric in metrics:
            if metric == 'f1_max':
                tables[dataset].loc[preproc, metric] = np.nanmax(
                    [tables[dataset].loc[preproc, metric], calculate_f1_max(run_info.artifact_uri)]
                )
                continue

            tables[dataset].loc[preproc, metric] = np.nanmax(
                [tables[dataset].loc[preproc, metric], run.data.metrics[metric]]
            )

    if ranks:
        ranks_by_dataset(tables, to_latex=True)

    for dataset, frame in tables.items():
        frame.index = preprocessings_pretty
        frame.columns = metrics_pretty

        if to_latex:
            styler = frame.style.format(na_rep='N/A', precision=3) \
                        .highlight_max(props='textbf:--rwrap;')

            with open(f'./thesis/tables/{dataset}_metrics.tex', 'w', encoding='utf-8') as table:
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
                    r'    \caption{}' '\n'
                    r'    \label{table:}' '\n'
                    r'\end{table}'
                )

                table.write(contents)

    return tables


def main(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_info = [
        run for run in mlflow.list_run_infos(experiment.experiment_id, max_results=10000)
        if run.status == 'FINISHED'
    ]

    datasets_stats(to_latex=True)
    preprocessings_configurations(to_latex=True)
    preprocessing_time(runs_info, to_latex=True)
    scores_by_dataset(runs_info, ranks=True, to_latex=True)


if __name__ == '__main__':
    main('http://127.0.0.1:5000', 'Benchmark')
