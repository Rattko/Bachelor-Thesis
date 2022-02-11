import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from core.exceptions import NoModelError
from core.metrics import precision, recall, f1, pr_auc, roc_auc, partial_roc_auc

class AutoML:
    """
    """

    def __init__(self, args, model, **params):
        """
        """

        self.args = args
        self.model = model
        self.models = None
        self.set_params(**params)

        # Initialize MLFlow
        mlflow.set_tracking_uri(self.args.tracking_uri)
        mlflow.set_experiment(self.args.experiment)
        mlflow.start_run(run_name=self.args.run_name)

        # Log AutoSklearn's hyperparameters into MLFlow
        mlflow.log_params(self.get_params())

    def fit(self, X, y):
        """
        """

        self.model.fit(X, y)
        self.model.refit(X, y)

        self.__get_models()

        # Pickle the trained AutoSklearn model into MLFlow
        mlflow.sklearn.log_model(
            self.model, 'model',
            registered_model_name=f'{self.args.experiment} - {self.args.run_name}',
            signature=mlflow.models.signature.infer_signature(X, pd.Series(y, dtype='float64')),
            input_example=X[:5]
        )

        # Log a number of trained models into MLFlow
        mlflow.log_metric('models_trained', len(self.models))

        # Log a model name, hyperparameters and metrics of all trained models into MLFlow
        self.__log_model_artifacts()

    def predict(self, X):
        """
        """

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        """

        return self.model.predict_proba(X)

    def score(self, X, y):
        """
        """

        preds = self.predict(X)
        preds_proba = self.predict_proba(X)

        precision_score = precision(y, preds)
        recall_score = recall(y, preds)
        f1_score = f1(y, preds)

        pr_auc_score = pr_auc(y, preds_proba)
        roc_auc_score = roc_auc(y, preds_proba)
        partial_roc_auc_score = partial_roc_auc(self.args.imbalance_ratio)(y, preds_proba)

        scores = [
            precision_score, recall_score, f1_score,
            pr_auc_score, roc_auc_score, partial_roc_auc_score
        ]

        # Log metrics of the best model obtained on the testing dataset into MLFlow
        mlflow.log_metrics(
            dict(zip(
                ['precision', 'recall', 'f1', 'pr_auc', 'roc_auc', 'partial_roc_auc'], scores
            ))
        )

        # Log ROC and PR curves obtained on the testing dataset into MLFlow
        self.__plot_pr_curve(y, preds_proba, pr_auc_score)
        self.__plot_roc_curve(y, preds_proba, roc_auc_score)
        self.__plot_roc_curve(y, preds_proba, roc_auc_score, max_fpr=0.25)

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
        try:
            config_id = self.model.leaderboard(include='config_id').iloc[0]['config_id']
            params = dict(self.model.automl_.runhistory_.ids_config[config_id])
        except KeyError as _:
            raise NoModelError('No model was trained within a given time limit') from None

        # Move the entry with a model which ended up in the ensemble to the very top
        results['pivot'] = range(1, len(results) + 1)
        results.loc[results['params'] == params, 'pivot'] = 0
        results = results.sort_values('pivot').drop('pivot', axis=1)

        results['hyperparams'] = self.__extract_hyperparams(results)
        results['cross_val_metrics'] = self.__extract_metrics(results)

        self.models = results[['classifier', 'hyperparams', 'cross_val_metrics']]

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

    def __plot_pr_curve(self, y, preds_proba, score):
        """
        """

        precision, recall, thresholds = precision_recall_curve(y, preds_proba[:, 1])

        fig = plt.figure()
        plt.plot(recall, precision, 'r-', label=f'AUC = {score:.3f}')

        plt.title('Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')

        plt.grid()

        mlflow.log_figure(fig, 'graph_pr_curve.png')
        mlflow.log_text('\n'.join(map(str, thresholds)), 'thresholds_pr_curve.txt')

    def __plot_roc_curve(self, y, preds_proba, score, max_fpr=None):
        """
        """

        curve_name = 'roc_curve' if max_fpr is None else 'partial_roc_curve'
        fpr, tpr, thresholds = roc_curve(y, preds_proba[:, 1])

        if max_fpr is not None:
            stop = np.searchsorted(fpr, max_fpr, 'right')
            tpr_at_max_fpr = np.interp(
                max_fpr, [fpr[stop - 1], fpr[stop]], [tpr[stop - 1], tpr[stop]]
            )
            threshold_at_max_fpr = np.interp(
                max_fpr, [fpr[stop - 1], fpr[stop]], [thresholds[stop - 1], thresholds[stop]]
            )

            fpr = np.append(fpr[:stop], max_fpr)
            tpr = np.append(tpr[:stop], tpr_at_max_fpr)
            thresholds = np.append(thresholds[:stop], threshold_at_max_fpr)

        fig = plt.figure()
        plt.plot(fpr, tpr, 'r-', label=f'AUC = {score:.3f}')
        plt.plot(
            np.linspace(0, fpr[-1], num=100),
            np.linspace(0, tpr[-1], num=100),
            'k-.', label=f'AUC = {0.5:.3f}'
        )

        plt.title('Receiver Operating Characteristic Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xscale('log')
        plt.legend(loc='lower right')

        plt.grid()

        mlflow.log_figure(fig, f'graph_{curve_name}.png')
        mlflow.log_text('\n'.join(map(str, thresholds)), f'thresholds_{curve_name}.txt')

    def __log_model_artifacts(self):
        """
        """

        n_models = len(self.models)
        for index, (_, model) in zip(range(n_models), self.models.iterrows()):
            mlflow.log_dict(dict(model), f'{index:0>{len(str(n_models))}}.json')