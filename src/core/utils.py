import os
import sys

import numpy as np


OVERSAMPLING_METHODS = {'random_oversampling', 'smote', 'svm_smote', 'k_means_smote', 'adasyn'}

def load_npz_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(f'./datasets/{path}/data_clean.npz') as data_file:
        data = data_file['X']
        target = data_file['y']

    return data[:10000], target[:10000]

def calculate_imbalance(labels: np.ndarray) -> float:
    return len(labels[labels == 1]) / len(labels[labels == 0])

def check_datasets(datasets: list[str]) -> list[str]:
    if datasets != ['all']:
        invalid_datasets = [
            dataset for dataset in datasets if dataset not in os.listdir('datasets')
        ]

        if invalid_datasets:
            print('These datasets are not available:', *invalid_datasets, file=sys.stderr)
            sys.exit(1)

    return os.listdir('datasets') if datasets == ['all'] else datasets

def check_preprocessings(preprocessings: list[str]) -> list[str]:
    if preprocessings == ['all']:
        return [
            preprocessing.removesuffix('.py')
            for preprocessing in os.listdir('src/core/preprocessing')
            if preprocessing != 'resampler.py'
        ]

    if preprocessings == ['oversampling']:
        return list(OVERSAMPLING_METHODS)

    if preprocessings == ['undersampling']:
        return list(
            {
                preprocessing.removesuffix('.py')
                for preprocessing in os.listdir('src/core/preprocessing')
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

def get_resampler_name(module_name: str) -> str:
    return module_name.title().replace('_', '') + 'Resampler'

def get_dataset_name(dataset_name: str) -> str:
    return f'Dataset {dataset_name.title()}'

def get_preproc_name(preproc_name: str) -> str:
    return preproc_name.title().replace('_', '')

def get_run_name(dataset_name: str, preproc_name: str) -> str:
    return f'{dataset_name} - {preproc_name}'
