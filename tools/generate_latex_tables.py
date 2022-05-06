#!/usr/bin/env python3

import importlib
import lzma
import os
import pickle
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.run_info import RunInfo

from core.utils import get_resampler_name


# Names of datasets ordered by the level of imbalance in descending order
datasets_pretty = [
    'Asteroid', 'Credit Card Subset', 'Credit Card', 'PC2', 'MC1', 'Employee Turnover',
    'Satellite', 'BNG - Solar Flare', 'Mammography', 'Letter', 'Relevant Images',
    'Click Prediction V1', 'Click Prediction V2', 'Amazon Employee', 'BNG - Sick',
    'Sylva Prior', 'BNG - Spect'
]

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
    'balanced_accuracy', 'precision', 'recall', 'f1', 'pr_auc',
    'matthews_corr_coef', 'roc_auc', 'partial_roc_auc'
]

metrics_pretty = [
    'B. Accuracy', 'Precision', 'Recall', 'F1', 'PR AUC', 'MCCoeff', 'ROC AUC', 'P-ROC AUC'
]


def datasets_stats(to_latex: bool = False) -> pd.DataFrame:
    stats = pd.DataFrame(columns=['name', 'size', 'majority_size', 'minority_size', 'imbalance'])

    for dataset_filename in os.listdir('./datasets/'):
        with lzma.open(f'./datasets/{dataset_filename}', 'rb') as dataset_file:
            dataset = pickle.load(dataset_file)

        stats.loc[len(stats)] = dataset.to_dict()

    stats.sort_values(by='imbalance', ascending=False, inplace=True)

    stats.columns = ['Name', 'Size', 'Majority Samples', 'Minority Samples', 'Imbalance']
    stats['name'] = datasets_pretty

    if to_latex:
        stats.to_latex('./thesis/tables/datasets.tex', index=False, label='table:dataset')

    return stats


def preprocessings_configurations(to_latex: bool = False) -> pd.DataFrame:
    configs = pd.DataFrame(columns=['Method', 'Hyperparameter Configurations'])

    for preprocessing, preprocessing_pretty in zip(preprocessings, preprocessings_pretty):
        preproc_module = importlib.import_module(f'core.preprocessings.{preprocessing}')
        resampler_cls = getattr(preproc_module, get_resampler_name(preprocessing))

        configs.loc[len(configs)] = [preprocessing_pretty, len(list(resampler_cls.hyperparams()))]

    if to_latex:
        configs.to_latex('./thesis/tables/configurations.tex', index=False, label='table:configs')

    return configs


def preprocessing_time(runs_info: list[RunInfo], to_latex: bool = False) -> pd.DataFrame:
    times = pd.DataFrame(index=preprocessings)

    for run_info in runs_info:
        run = mlflow.get_run(run_info.run_id)
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        if 'preprocessing_time' in run.data.metrics.keys():
            times.loc[preproc, dataset] = run.data.metrics['preprocessing_time']
        else:
            times.loc[preproc, dataset] = 0.0

    times.fillna(value='N/A', inplace=True)

    if to_latex:
        times.to_latex('./thesis/tables/preprocessing_time.tex')

    return times


def scores_by_dataset(
    runs_info: list[RunInfo], to_latex: bool = False
) -> defaultdict[str, pd.DataFrame]:
    tables: defaultdict[str, pd.DataFrame] = defaultdict(
        lambda: pd.DataFrame(index=preprocessings, columns=metrics)
    )

    for run_info in runs_info:
        run = mlflow.get_run(run_info.run_id)
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        for metric in metrics:
            tables[dataset].loc[preproc, metric] = np.nanmax(
                [tables[dataset].loc[preproc, metric], run.data.metrics[metric]]
            )

    for dataset, frame in tables.items():
        frame.index = preprocessings_pretty
        frame.columns = metrics_pretty

        # Round all values to 3 decimal places and replace NaNs
        frame = frame.apply(lambda x: np.around(x.astype(float), 3)).fillna('N/A')

        if to_latex:
            with open(f'./thesis/tables/{dataset}_metrics.tex', 'w', encoding='utf-8') as table:
                contents = (
                    r'\clearpage' '\n'
                    r'\begin{table}' '\n'
                    r'    \centering' '\n'
                    r'    \widesplit{' '\n'
                    r'        \makebox[\textwidth]{' '\n'
                    r'            \begin{tabularx}{\textwidth}{lRRRR}' '\n'
                    f'                {frame.to_latex()}'
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
    scores_by_dataset(runs_info, to_latex=True)


if __name__ == '__main__':
    main('http://127.0.0.1:5000', 'Benchmark')
