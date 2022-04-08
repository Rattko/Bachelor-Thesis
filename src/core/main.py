#!/usr/bin/env python3

import argparse
import importlib

import autosklearn.pipeline.components.data_preprocessing
import mlflow
import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split

from core.automl import AutoML
from core.logger import Logger
from core.metrics import accuracy, balanced_accuracy, f1, log_loss, matthews_corr_coef
from core.metrics import partial_roc_auc, precision, pr_auc, recall, roc_auc
from core.no_preprocessing import NoPreprocessing
from core.utils import calculate_imbalance, check_datasets, check_preprocessings
from core.utils import get_dataset_name, get_preproc_name, get_resampler_name, get_run_name
from core.utils import load_npz_dataset


parser = argparse.ArgumentParser()

# MLFlow related switches
parser.add_argument('--tracking_uri', type=str, default='http://127.0.0.1:5000')
parser.add_argument('--experiment', type=str, default='Benchmark')

# AutoSklearn related switches
parser.add_argument('--total_time', type=int, default=30)
parser.add_argument('--time_per_run', type=int, default=10)
parser.add_argument('--memory', type=int, default=None)

# Common switches
parser.add_argument('--test_size', type=float, default=0.25)
parser.add_argument('--random_state', type=int, default=1)

parser.add_argument('--datasets', type=str, nargs='+', default=['all'])
parser.add_argument('--preprocessings', type=str, nargs='+', default=['all'])
parser.add_argument('--grid_search', action='store_true', default=False)

def run_experiment(
    args: argparse.Namespace, logger: Logger,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> None:
    imbalance_ratio = calculate_imbalance(train_target)

    automl = AutoML(
        imbalance_ratio, AutoSklearnClassifier(), logger, **{
            'time_left_for_this_task': args.total_time, # In seconds
            'per_run_time_limit': args.time_per_run, # In seconds
            'memory_limit': args.memory, # In megabytes
            'include': {
                'data_preprocessor': ['NoPreprocessing'],
                'feature_preprocessor': ['no_preprocessing']
            },
            'exclude': {
                'classifier': ['mlp']
            },
            'resampling_strategy': 'cv',
            'resampling_strategy_arguments': {'folds': 5},
            'metric': roc_auc, # Metric used to evaluate and produce the final ensemble
            'scoring_functions': [
                accuracy, balanced_accuracy, precision, recall, f1, log_loss,
                pr_auc, matthews_corr_coef, roc_auc, partial_roc_auc(imbalance_ratio)
            ],
            'ensemble_size': 1,
            'ensemble_nbest': 1,
            'initial_configurations_via_metalearning': 0,
            'n_jobs': -1,
            'seed': args.random_state
        }
    )

    automl.fit(train_data, train_target)
    automl.score(test_data, test_target)

def main(args: argparse.Namespace) -> None:
    # Check and parse datasets and preprocessings given on the command line
    args.datasets = check_datasets(args.datasets)
    args.preprocessings = check_preprocessings(args.preprocessings)

    # Initialize MLFlow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    for dataset in args.datasets:
        data, target = load_npz_dataset(dataset)
        train_data, test_data, train_target, test_target = train_test_split(
            data, target, stratify=target, test_size=args.test_size, random_state=args.random_state
        )

        for preprocessing in args.preprocessings:
            # Retrieve the correct preprocessing module
            preproc_module = importlib.import_module(f'core.preprocessings.{preprocessing}')
            resampler_cls = getattr(preproc_module, get_resampler_name(preprocessing))

            dataset_name = get_dataset_name(dataset)
            preproc_name = get_preproc_name(preprocessing)

            logger = Logger(dataset_name, preproc_name)

            if not args.grid_search:
                with mlflow.start_run(run_name=get_run_name(dataset_name, preproc_name)):
                    # Resample the training part of the dataset
                    resampler = resampler_cls(logger, random_state=args.random_state)

                    try:
                        preproc_data = resampler.fit_resample(train_data, train_target)
                        run_experiment(
                            args, logger, *preproc_data, test_data, test_target
                        )
                    except Exception as exc:
                        mlflow.end_run(status='FAILED')
                        print(exc)

                continue

            for preproc_hyperparam_config in resampler_cls.hyperparams():
                with mlflow.start_run(run_name=get_run_name(dataset_name, preproc_name)):
                    # Resample the training part of the dataset
                    resampler = resampler_cls(
                        logger, **preproc_hyperparam_config, random_state=args.random_state
                    )

                    try:
                        preproc_data = resampler.fit_resample(train_data, train_target)
                        run_experiment(
                            args, logger, *preproc_data, test_data, test_target
                        )
                    except Exception as exc:
                        mlflow.end_run(status='FAILED')
                        print(exc)

if __name__ == '__main__':
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)
    args = parser.parse_args()
    main(args)
