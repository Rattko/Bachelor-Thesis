#!/usr/bin/env python3

import os
from collections import defaultdict

import mlflow
import numpy as np
import pandas as pd


def scores_by_dataset(experiment_name: str) -> defaultdict[str, pd.DataFrame]:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_info = [
        run for run in mlflow.list_run_infos(experiment.experiment_id, max_results=100000)
        if run.status == 'FINISHED'
    ]

    preprocessings = [
        preproc.removesuffix('.py')
        for preproc in os.listdir('./src/core/preprocessings/')
        if preproc != 'resampler.py'
    ]

    metrics = [
        'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'log_loss',
        'pr_auc', 'matthews_corr_coef', 'roc_auc', 'partial_roc_auc'
    ]

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

    preprocessings = [
        'Baseline', 'Random Oversampling', 'SMOTE', 'Borderline SMOTE', 'SVM SMOTE',
        'KMeans SMOTE', 'ADASYN', 'Random Undersampling', 'CNN', 'ENN', 'Repeated ENN',
        'All KNN', 'Near Miss', 'Tomek Links', 'One-Sided Selection', 'NCL', 'Cluster Centroids'
    ]

    metrics = [
        'Accuracy', 'B. Accuracy', 'Precision', 'Recall', 'F1', 'Log Loss',
        'PR AUC', 'M. Corr. Coeff.', 'ROC AUC', 'PROC AUC'
    ]

    for dataset, df in tables.items():
        df.index = preprocessings
        df.columns = metrics
        df.fillna('N/A', inplace=True)

        df.to_latex(f'./thesis/tables/{dataset}_metrics.tex')

    return tables


def main(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)

    scores_by_dataset(experiment_name)


if __name__ == '__main__':
    main('http://127.0.0.1:5000', 'Benchmark')
