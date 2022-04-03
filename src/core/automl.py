""" Module implementing a wrapper over AutoSklearn. """

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autosklearn.estimators import AutoSklearnClassifier
from autosklearn.metrics import _PredictScorer, _ProbaScorer, _ThresholdScorer
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from core.exceptions import NoModelError
from core.logger import Logger


class AutoML:
    """ Wrapper over AutoSklearn with additional functionality.

    This class follows the standard interface of estimators from ScikitLearn. It uses
    AutoSklearn for training and prediction and adds the ability to extract meaningful
    information from AutoSklearn easily. It also allows for dependency injection to log
    information, such as trained models, their hyperparameters and cross-validation scores,
    scores, graphs, etc.

    Attributes
    ----------
    imbalance_ratio : float
        Ratio between the size of the minority class and the size of the majority class.
    model : AutoSklearnClassifier
        Instance of `AutoSklearnClassifier`.
    models : pd.DataFrame
        Dataframe containing all trained models and information about them.
    __logger : Logger
        Instance of `Logger`.
    """

    def __init__(
        self, imbalance_ratio: float, model: AutoSklearnClassifier, logger: Logger, **params: Any
    ) -> None:
        """
        Parameters
        ----------
        imbalance_ratio : float
            Ratio between the size of the minority class and the size of the majority class.
        model : AutoSklearnClassifier
            Instance of `AutoSklearnClassifier`.
        logger : Logger
            Instance of `Logger`.
        **params : dict[str, Any]
            Hyperparameters for the `AutoSklearnClassifier`.
        """

        self.imbalance_ratio = imbalance_ratio

        self.model = model
        self.models = None
        self.set_params(**params)

        self.__logger = logger
        self.__logger.log_params('autosklearn', self.get_params())

    def fit(self, data: np.ndarray, target: np.ndarray) -> None:
        """ Fit AutoSklearn model to the given data.

        This method also automatically refits all found models on the whole training data.

        Parameters
        ----------
        data : np.ndarray
            Matrix containing training data of shape (n_samples, n_features).
        target : np.ndarray
            Vector containing target values of shape (n_samples,).

        See Also
        --------
        The documentation [1] for AutoSklearn to find more details on
        algorithms used under the hood.

        [1]: https://automl.github.io/auto-sklearn/master
        """

        self.model.fit(data, target)
        self.model.refit(data, target)

        self.__get_models()

        self.__logger.log_model(self.model)
        self.__logger.log_metrics(len(self.models))
        self.__logger.log_artifacts(self.models)

    def predict(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        """ Predict class labels for the given data.

        Parameters
        ----------
        data : np.ndarray
            Matrix containing testing data of shape (n_samples, n_features).
        target : Optional[np.ndarray]
            Vector containing target values of shape (n_samples,).

        Returns
        -------
        preds : np.ndarray
            Vector containing predictions for the given data of shape (n_samples,).
        """

        preds = self.model.predict(data)

        self.__logger.log_preds(data, target, preds)

        return preds

    def predict_proba(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> np.ndarray:
        """ Predict the probability of belonging to each class for the given data.

        Parameters
        ----------
        data : np.ndarray
            Matrix containing testing data of shape (n_samples, n_features).
        target : Optional[np.ndarray]
            Vector containing target values of shape (n_samples,).

        Returns
        -------
        preds_proba : np.ndarray
            Matrix containing predictions for the given data of shape (n_samples, n_classes).
        """

        preds_proba = self.model.predict_proba(data)

        self.__logger.log_preds(data, target, preds_proba)

        return preds_proba

    def score(self, data: np.ndarray, target: np.ndarray) -> dict[str, float]:
        """ Compute metrics to evaluate model performance.

        Parameters
        ----------
        data : np.ndarray
            Matrix containing testing data of shape (n_samples, n_features).
        target : np.ndarray
            Vector containing target values of shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If `score_func` is not an instance of `_PredictScorer`, `_ThresholdScorer`
            nor `_ProbaScorer`. This exception should never occur.

        Returns
        -------
        scores : dict[str, float]
            Dictionary containing names and values of metrics obtained on the given data.
        """

        preds = self.predict(data, target)
        preds_proba = self.predict_proba(data, target)

        scores = {}
        for score_func in self.model.scoring_functions:
            if isinstance(score_func, _PredictScorer):
                scores[score_func.name] = score_func(target, preds)
            elif isinstance(score_func, _ThresholdScorer | _ProbaScorer):
                scores[score_func.name] = score_func(target, preds_proba)
            else:
                raise NotImplementedError()

        self.__logger.log_metrics(scores)

        self.plot_pr_curve(target, preds_proba, scores['pr_auc'])
        self.plot_roc_curve(target, preds_proba, scores['roc_auc'])
        self.plot_roc_curve(target, preds_proba, scores['partial_roc_auc'], self.imbalance_ratio)

        return scores

    def plot_pr_curve(self, target: np.ndarray, preds_proba: np.ndarray, score: float) -> Figure:
        """ Plot the PR curve using the given data.

        Parameters
        ----------
        target : np.ndarray
            Vector containing target values of shape (n_samples,).
        preds_proba : np.ndarray
            Matrix containing predictions of shape (n_samples, n_classes).
        score : float
            Area under the PR curve obtained on the given data.

        Returns
        -------
        fig : Figure
            Figure instance containing the plot of the PR curve.
        """

        precision, recall, thresholds = precision_recall_curve(target, preds_proba[:, 1])

        fig = plt.figure()
        plt.plot(recall, precision, 'r-', label=f'AUC = {score:.3f}')

        plt.title('Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')

        plt.grid()

        self.__logger.log_curve(fig, *(precision, recall, thresholds))

        return fig

    def plot_roc_curve(
        self, target: np.ndarray, preds_proba: np.ndarray,
        score: float, max_fpr: Optional[float] = None
    ) -> Figure:
        """ Plot the ROC curve using the given data.

        If the `max_fpr` parameter is provided, the created plot will contain
        the partial ROC curve on an interval [0, `max_fpr`].

        Parameters
        ----------
        target : np.ndarray
            Vector containing target values of shape (n_samples,).
        preds_proba : np.ndarray
            Matrix containing predictions of shape (n_samples, n_classes).
        score : float
            Area under the ROC curve obtained on the given data.
        max_fpr : Optional[float]
            Upper bound for the FPR values, i.e. [0, max_fpr].

        Returns
        -------
        fig : Figure
            Figure instance containing the plot of the ROC curve.
        """

        fpr, tpr, thresholds = roc_curve(target, preds_proba[:, 1])

        if max_fpr is not None:
            stop = int(np.searchsorted(fpr, max_fpr, 'right'))
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
            'k-.', label='AUC = {0.5:.3f}'
        )

        plt.title('Receiver Operating Characteristic Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xscale('log')
        plt.legend(loc='lower right')

        plt.grid()

        self.__logger.log_curve(fig, *(fpr, tpr, thresholds))

        return fig

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """ Get hyperparameters of the AutoSklearnClassifier.

        Parameters
        ----------
        deep : bool
            If true, get hyperparameters of this model and nested models as well.

        Returns
        -------
        params : dict[str, Any]
            Hyperparameter names mapped to their values.
        """

        return self.model.get_params(deep)

    def set_params(self, **params: Any) -> None:
        """ Set hyperparameters of the AutoSklearnClassifier.

        Parameters
        ----------
        **params : dict[str, Any]
            AutoSklearnClassifier's hyperparameters.
        """

        self.model.set_params(**params)

    def __get_models(self) -> None:
        """ Extract information about trained models from AutoSklearn.

        Extract names of successfully trained classifiers, their hyperparameters and
        cross-validation metrics from AutoSklearn. The single best model which ends up
        in the ensemble is at the first position of the dataframe.

        Raises
        ------
        NoModelError
            If AutoSklearn could not fully train any model with the given resources.
        """

        results = pd.DataFrame.from_dict(self.model.cv_results_).set_index('rank_test_scores')
        results = results.rename({'param_classifier:__choice__': 'classifier'}, axis=1)
        results = results[results['status'] == 'Success'].sort_index()

        # Get `config_id` of a model which ended up in the ensemble and then get its hyperparams
        try:
            config_id = self.model.leaderboard(include='config_id').iloc[0]['config_id']
            params = dict(self.model.automl_.runhistory_.ids_config[config_id])
        except KeyError:
            raise NoModelError('No model was trained within a given time limit') from None

        # Move the entry with a model which ended up in the ensemble to the very top
        results['pivot'] = range(1, len(results) + 1)
        results.loc[results['params'] == params, 'pivot'] = 0
        results = results.sort_values('pivot').drop('pivot', axis=1)

        results['hyperparams'] = self.__extract_hyperparams(results)
        results['cross_val_metrics'] = self.__extract_metrics(results)

        self.models = results[['classifier', 'hyperparams', 'cross_val_metrics']]

    def __extract_hyperparams(self, results: pd.DataFrame) -> list[dict[str, Any]]:
        """ Extract hyperparameters related to classifiers.

        AutoSklearn provides information about data and features preprocessing methods
        that we do not need. We filter out these pieces of information and only keep
        the ones related to classifiers.

        Parameters
        ----------
        results : pd.DataFrame
            Information about the training provided by AutoSklearn.

        Returns
        -------
        hyperparams : list[dict[str, Any]]
            Hyperparameters related to classifiers.
        """

        return [
            {
                key.split(':')[2]: value
                for key, value in row.items()
                if key.startswith('classifier:') and not key.endswith('__choice__')
            } for row in results['params']
        ]

    def __extract_metrics(self, results: pd.DataFrame) -> list[dict[str, float]]:
        """ Extract cross-validation metrics obtained during the training.

        Parameters
        ----------
        results : pd.DataFrame
            Information about the training provided by AutoSklearn.

        Returns
        -------
        metrics : list[dict[str, float]]
            Cross-validation metrics obtained during the training.
        """

        metric_cols = [col for col in results.columns if col.startswith('metric_')]

        return [
            row.rename(lambda x: x.replace('metric_', '')).to_dict()
            for _, row in results[metric_cols].iterrows()
        ]
