#!/usr/bin/env python3

import argparse
import importlib
import random
import socket
import traceback
from typing import Any, Optional

import autosklearn.pipeline.components.data_preprocessing
import mlflow
import numpy as np
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import train_test_split

from core.automl import AutoML
from core.dataset import Dataset
from core.exceptions import NoModelError
from core.logger import Logger
from core.metrics import accuracy, balanced_accuracy, f1, log_loss, matthews_corr_coef
from core.metrics import partial_roc_auc, precision, pr_auc, recall, roc_auc
from core.no_preprocessing import NoPreprocessing
from core.preprocessings.resampler import Resampler
from core.utils import calculate_imbalance, check_datasets, check_preprocessings
from core.utils import get_preproc_name, get_resampler_name, get_run_name
from core.utils import load_dataset


parser = argparse.ArgumentParser()

# Mlflow related switches
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


def experiment_running_or_finished(
    args: argparse.Namespace, dataset_id: int, preprocessing: str, hyperparams: dict[str, Any]
) -> bool:
    hyperparams = {key: str(value) for key, value in hyperparams.items()}

    experiment = mlflow.get_experiment_by_name(args.experiment)
    runs_info = [
        run for run in mlflow.list_run_infos(experiment.experiment_id)
        if run.status in ('FINISHED', 'RUNNING')
    ]

    runs_to_skip = []
    for run_info in runs_info:
        run = mlflow.get_run(run_info.run_id)
        preproc_hyperparams = {
            param.removeprefix('imblearn__'): value
            for param, value in run.data.params.items()
            if param.startswith('imblearn')
        }

        runs_to_skip.append(
            [(run.data.tags['dataset'], run.data.tags['preprocessing']), preproc_hyperparams]
        )

    return any(
        (str(dataset_id), preprocessing) == names and params | hyperparams == params
        for names, params in runs_to_skip
    )


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


def initialise_run(
    args: argparse.Namespace, dataset: Dataset, resampler_cls: Resampler, preprocessing: str,
    train_data: np.ndarray, train_target: np.ndarray, test_data: np.ndarray,
    test_target: np.ndarray, preproc_hyperparams: Optional[dict[str, Any]] = None
) -> None:
    mlflow.start_run(run_name=get_run_name(dataset.dataset_id, get_preproc_name(preprocessing)))
    logger = Logger(args.experiment, dataset.name, get_preproc_name(preprocessing))

    logger.set_tags({
        'machine': socket.gethostname(),
        'dataset': dataset.dataset_id,
        'preprocessing': preprocessing
    })
    logger.log_params('dataset', dataset.to_dict())

    # Resample the training part of the dataset
    if preproc_hyperparams is None:
        resampler = resampler_cls(logger, random_state=args.random_state)
    else:
        resampler = resampler_cls(logger, **preproc_hyperparams, random_state=args.random_state)

    try:
        preproc_data = resampler.fit_resample(train_data, train_target)
        run_experiment(args, logger, *preproc_data, test_data, test_target)
        mlflow.end_run()
    except NoModelError:
        logger.set_tags({'no_model_error': True})
    except Exception as exc:
        traceback.print_exception(exc)
    finally:
        # Handle cases where execution have failed and thus have disallowed to end a run
        # successfully. This includes an occurence of any exception or halting with Ctrl+C
        # for example.
        mlflow.end_run(status='FAILED')


def main(args: argparse.Namespace) -> None:
    # Check and parse datasets and preprocessings given on the command line
    args.datasets = check_datasets(args.datasets)
    args.preprocessings = check_preprocessings(args.preprocessings)

    # Initialize MLFlow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    for dataset in random.sample(args.datasets, len(args.datasets)):
        dataset = load_dataset(dataset)
        train_data, test_data, train_target, test_target = train_test_split(
            dataset.data, dataset.target, stratify=dataset.target,
            test_size=args.test_size, random_state=args.random_state
        )

        for preprocessing in random.sample(args.preprocessings, len(args.preprocessings)):
            # Retrieve the correct preprocessing module
            preproc_module = importlib.import_module(f'core.preprocessings.{preprocessing}')
            resampler_cls = getattr(preproc_module, get_resampler_name(preprocessing))

            if not args.grid_search:
                initialise_run(
                    args, dataset, resampler_cls, preprocessing,
                    train_data, train_target, test_data, test_target
                )

                continue

            for preproc_hyperparams in resampler_cls.hyperparams():
                if experiment_running_or_finished(
                    args, dataset.dataset_id, preprocessing, preproc_hyperparams
                ):
                    continue

                initialise_run(
                    args, dataset, resampler_cls, preprocessing, train_data,
                    train_target, test_data, test_target, preproc_hyperparams
                )


if __name__ == '__main__':
    autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing)
    args = parser.parse_args()
    main(args)
