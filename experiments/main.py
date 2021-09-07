from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score

from autosklearn.metrics import precision, recall, f1, roc_auc, make_scorer
from autosklearn.classification import AutoSklearnClassifier

import pandas as pd
import mlflow

# TODO: Add documentation comments
# TODO: Baseline for base datasets
# TODO: Basic data preprocessing
# TODO: Pickle the best model
# TODO: Log graphs of ROC and PR curves
# TODO: Remove preprocessing steps from AutoSklearn
# TODO: Update AutoSklearn's hyperparameters
# TODO: Intel Acceleration
class AutoML:
    """
    """

    def __init__(self, model, **params):
        """
        """

        self.model = model
        self.models = None
        self.set_params(**params)

        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.set_experiment('Breast Cancer Dataset')
        mlflow.start_run(run_name='Baseline')

        mlflow.log_params(self.get_params())

    def fit(self, X, y):
        """
        """

        self.model.fit(X, y)
        self.__get_models()

        mlflow.log_metrics(self.models.iloc[0]['metrics'])
        self.__log_artifacts()

    def predict(self, X):
        """
        """

        return self.model.predict(X)

    def score(self, X, y):
        """
        """

        preds = self.predict(X)

        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds)
        recall = recall_score(y, preds)

        return [accuracy, precision, recall]

    def get_params(self):
        """
        """

        return self.model.get_params()

    def set_params(self, **params):
        """
        """

        self.model.set_params(**params)

    def __get_models(self):
        """
        """

        results = pd.DataFrame.from_dict(self.model.cv_results_).set_index('rank_test_scores')
        results = results.rename({'param_classifier:__choice__': 'classifier'}, axis=1)
        results = results[results['status'] == 'Success'].sort_index()

        # Get 'config_id' of a model which ended up in the ensemble and then get its hyperparams
        config_id = self.model.leaderboard(include='config_id').iloc[0]['config_id']
        params = dict(self.model.automl_.runhistory_.ids_config[config_id])

        # Move the entry with a model which ended up in the ensemble to the very top
        results['pivot'] = range(1, len(results) + 1)
        results.loc[results['params'] == params, 'pivot'] = 0
        results = results.sort_values('pivot').drop('pivot', axis=1)

        results['hyperparams'] = self.__extract_hyperparams(results)
        results['metrics'] = self.__extract_metrics(results)

        self.models = results[['classifier', 'hyperparams', 'metrics']]

    def __extract_hyperparams(self, results):
        """
        """

        return [
            {
                key.split(':')[2] : value
                for key, value in row.items()
                if key.startswith('classifier:') and not key.endswith('__choice__')
            } for row in results['params']
        ]

    def __extract_metrics(self, results):
        """
        """

        metric_cols = [col for col in results.columns if col.startswith('metric_')]
        return [
            row.rename(lambda x: x.replace('metric_', '')).to_dict()
            for _, row in results[metric_cols].iterrows()
        ]

    def __log_artifacts(self):
        """
        """

        n_models = len(self.models)
        for index, (_, model) in zip(range(n_models), self.models.iterrows()):
            mlflow.log_dict(dict(model), f'{index:0>{len(str(n_models))}}.json')


def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=1
    )

    pr_auc = make_scorer(
        name='pr_auc',
        score_func=average_precision_score,
        greater_is_better=True,
        needs_threshold=True
    )

    partial_roc_auc = make_scorer(
        name='partial_roc_auc',
        score_func=roc_auc_score,
        greater_is_better=True,
        needs_threshold=True,
        **{'max_fpr': 0.25}
    )

    automl = AutoML(
        AutoSklearnClassifier(), **{
            'time_left_for_this_task': 30, # In seconds
            'per_run_time_limit': 10, # In seconds
            'memory_limit': 4096, # In megabytes
            'include_preprocessors': ['no_preprocessing'],
            'exclude_estimators': ['mlp'],
            'metric': roc_auc, # Metric used to evaluate and produce the final ensemble
            'scoring_functions': [precision, recall, f1, roc_auc, partial_roc_auc, pr_auc],
            'ensemble_size': 1,
            'ensemble_nbest': 1,
            'initial_configurations_via_metalearning': 0,
            'n_jobs': -1,
            'seed': 1
        }
    )

    automl.fit(X_train, y_train)
    print(automl.score(X_test, y_test))

if __name__ == '__main__':
    main()
