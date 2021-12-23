from argparse import ArgumentParser

import mlflow
from autosklearn.classification import AutoSklearnClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from automl import AutoML
from metrics import precision, recall, f1, pr_auc, roc_auc, partial_roc_auc

# TODO: Add documentation comments
# TODO: Baseline for base datasets
# TODO: Basic data preprocessing
# TODO: Remove preprocessing steps from AutoSklearn
# TODO: Intel Acceleration
# TODO: Split AutoSklearn and MLFlow functionality using decorators

parser = ArgumentParser()

# MLFlow related switches
parser.add_argument('--tracking_uri', type=str, default='http://127.0.0.1:5000')
parser.add_argument('--experiment', type=str, default='Testing Dataset')
parser.add_argument('--run_name', type=str, default='Test Run')

# AutoSklearn related switches
parser.add_argument('--total_time', type=int, default=30)
parser.add_argument('--time_per_run', type=int, default=10)
parser.add_argument('--memory', type=int, default=None)

# Common switches
parser.add_argument('--dataset_path', type=str, default=None)
parser.add_argument('--random_state', type=int, default=1)

def main(args):
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.random_state
    )

    automl = AutoML(
        args, AutoSklearnClassifier(), **{
            'time_left_for_this_task': args.total_time, # In seconds
            'per_run_time_limit': args.time_per_run, # In seconds
            'memory_limit': args.memory, # In megabytes
            'include': {
                'feature_preprocessor': ['no_preprocessing']
            },
            'exclude': {
                'classifier': ['mlp']
            },
            'resampling_strategy': 'cv',
            'resampling_strategy_arguments': {'folds': 5},
            'metric': roc_auc, # Metric used to evaluate and produce the final ensemble
            'scoring_functions': [
                precision, recall, f1, pr_auc, roc_auc, partial_roc_auc
            ],
            'ensemble_size': 1,
            'ensemble_nbest': 1,
            'initial_configurations_via_metalearning': 0,
            'n_jobs': -1,
            'seed': args.random_state
        }
    )

    automl.fit(X_train, y_train)
    automl.score(X_test, y_test)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
