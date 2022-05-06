import lzma
import os
import pickle
import sys
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from core.dataset import Dataset


OVERSAMPLING_METHODS = {
    'random_oversampling', 'smote', 'borderline_smote', 'svm_smote', 'k_means_smote', 'adasyn'
}


def load_dataset(path: str) -> Dataset | tuple[np.ndarray, np.ndarray]:
    if os.path.isdir(path):
        return load_npz_dataset(path)

    return load_pickle_dataset(path)


def load_pickle_dataset(path: str) -> Dataset:
    with lzma.open(f'./datasets/{path}.pickle', 'rb') as data_file:
        dataset = pickle.load(data_file)

    return dataset


def load_npz_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(f'./datasets/{path}/data_clean.npz') as data_file:
        data = data_file['X']
        target = data_file['y']

    return data, target


def check_datasets(datasets: list[str]) -> list[str]:
    available_datasets = [dataset.removesuffix('.pickle') for dataset in os.listdir('datasets')]

    if datasets != ['all']:
        invalid_datasets = [
            dataset for dataset in datasets if dataset not in available_datasets
        ]

        if invalid_datasets:
            print('These datasets are not available:', *invalid_datasets, file=sys.stderr)
            sys.exit(1)

    return available_datasets if datasets == ['all'] else datasets


def check_preprocessings(preprocessings: list[str]) -> list[str]:
    if preprocessings == ['all']:
        return [
            preprocessing.removesuffix('.py')
            for preprocessing in os.listdir('src/core/preprocessings')
            if preprocessing != 'resampler.py'
        ]

    if preprocessings == ['oversampling']:
        return list(OVERSAMPLING_METHODS)

    if preprocessings == ['undersampling']:
        return list(
            {
                preprocessing.removesuffix('.py')
                for preprocessing in os.listdir('src/core/preprocessings')
                if preprocessing != 'resampler.py'
            } - OVERSAMPLING_METHODS
        )

    invalid_preprocessings = [
        preprocessing for preprocessing in preprocessings
        if f'{preprocessing}.py' not in os.listdir('src/core/preprocessings')
    ]

    if invalid_preprocessings:
        print('These preprocessings are not available:', *invalid_preprocessings, file=sys.stderr)
        sys.exit(1)

    return [preproc.removesuffix('.py') for preproc in preprocessings]


def get_runs_status(
    tracking_uri: str, experiment_name: str, return_value: Optional[str] = 'all'
) -> pd.DataFrame | list[tuple[str, str]] | None:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_info = mlflow.list_run_infos(experiment.experiment_id, max_results=100000)

    df = pd.DataFrame()

    for run_info in runs_info:
        run = mlflow.get_run(run_info.run_id)
        dataset = run.data.tags['dataset']
        preproc = run.data.tags['preprocessing']

        df.loc[preproc, dataset] = run_info.status

    df.fillna(value='NOT ATTEMPTED', inplace=True)

    if return_value == 'all':
        return df

    if return_value in ['finished', 'failed', 'not attempted']:
        rows, cols = np.where(df.values == return_value.upper())

        return [(df.index[row], df.columns[col]) for row, col in zip(rows, cols)]

    return None


def calculate_imbalance(labels: np.ndarray) -> float:
    return len(labels[labels == 1]) / len(labels[labels == 0])


def get_resampler_name(module_name: str) -> str:
    return module_name.title().replace('_', '') + 'Resampler'


def get_preproc_name(preproc_name: str) -> str:
    return preproc_name.title().replace('_', '')


def get_run_name(dataset_name: str, preproc_name: str) -> str:
    return f'{preproc_name} - {dataset_name}'


def get_model_name(experiment: str, dataset_name: str, preproc_name: str) -> str:
    return f'{experiment} / {preproc_name} - {dataset_name}'
