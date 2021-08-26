from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score

from autosklearn.metrics import precision, recall, f1, roc_auc, make_scorer
from autosklearn.classification import AutoSklearnClassifier

import pandas as pd

# TODO: Add documentation comments
class AutoML:
    """
    """

    def __init__(self, model, **params):
        """
        """

        self.model = model
        self.models = None
        self.set_params(**params)

    def fit(self, X, y):
        """
        """

        self.model.fit(X, y)
        self.__get_models()

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

        results['hyperparams'] = self.__extract_hyperparams(results)

        cols = ['classifier', 'hyperparams']
        cols.extend([key for key in results.keys() if key.startswith('metric_')])

        self.models = results[cols]

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


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    pr_auc = make_scorer(name='pr_auc', score_func=average_precision_score)

    automl = AutoML(
        AutoSklearnClassifier(), **{
            'scoring_functions': [precision, recall, f1, roc_auc, pr_auc],
            'time_left_for_this_task': 30,
            'per_run_time_limit': 10,
            'ensemble_size': 1,
            'n_jobs': -1
        }
    )

    automl.fit(X_train, y_train)
    print(automl.score(X_test, y_test))

if __name__ == '__main__':
    main()
