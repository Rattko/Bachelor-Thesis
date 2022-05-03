#!/usr/bin/env python3

import os

import mlflow
import numpy as np
import pandas as pd


def main(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_info = [
        run for run in mlflow.list_run_infos(experiment.experiment_id, max_results=100000)
        if run.status == 'FINISHED'
    ]

    preprocessings = [
        os.path.splitext(preproc)[0]
        for preproc in os.listdir('./src/core/preprocessings/')
        if preproc != 'resampler.py'
    ]
    metrics = [
        'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'log_loss',
        'pr_auc', 'matthews_corr_coef', 'roc_auc', 'partial_roc_auc'
    ]

    df = pd.DataFrame(index=preprocessings)
    tables = {metric: df.copy() for metric in metrics}

    for run_info in runs_info:
        run = mlflow.get_run(run_info.run_id)
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        for metric, value in run.data.metrics.items():
            if metric in ('preprocessing_time', 'models_trained'):
                continue

            if dataset in tables[metric].columns:
                tables[metric].loc[preproc, dataset] = np.nanmax(
                    [tables[metric].loc[preproc, dataset], value]
                )
            else:
                tables[metric].loc[preproc, dataset] = value

    for metric, table in tables.items():
        table.fillna(value='x', inplace=True)
        table.to_html(f'./tables/{metric}.html')


if __name__ == '__main__':
    main('http://127.0.0.1:5000', 'Benchmark')
