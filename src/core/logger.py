import mlflow

from functools import wraps
from timeit import default_timer as timer
from typing import Any

from core.utils import get_model_name


class Logger:
    def __init__(self, experiment: str, dataset_name: str, preproc_name: str) -> None:
        self.experiment = experiment
        self.dataset_name = dataset_name
        self.preproc_name = preproc_name

    def log_params(self, prefix: str, params: dict[str, Any]) -> None:
        # Log namespaced hyperparameters into MLFlow
        mlflow.log_params({f'{prefix}__{key}': value for key, value in params.items()})

    def log_model(self, model) -> None:
        # Pickle the trained AutoSklearn model into MLFlow
        model_name = get_model_name(self.experiment, self.dataset_name, self.preproc_name)
        mlflow.sklearn.log_model(model, 'model', registered_model_name=model_name)

    def log_preds(self, data, target, preds) -> None:
        file_name = 'preds.json' if len(preds.shape) == 1 else 'preds_proba.json'
        file_contents = [{
            'data_sample': list(x),
            'target': int(y),
            'preds': int(pred) if len(preds.shape) == 1 else list(pred)
        } for x, y, pred in zip(data, target, preds)]

        mlflow.log_dict(file_contents, file_name)

    def log_metrics(self, metric) -> None:
        if isinstance(metric, dict):
            # Log metrics of the best model obtained on the testing dataset into MLFlow
            mlflow.log_metrics(metric)
        elif isinstance(metric, int):
            # Log a number of trained models into MLFlow
            mlflow.log_metric('models_trained', metric)
        else:
            raise NotImplementedError()

    def log_curve(self, figure, *curve_data) -> None:
        title = figure.axes[0].get_title()

        if title == 'Precision Recall Curve':
            self.__log_pr_curve(figure, *curve_data)
        elif title == 'Receiver Operating Characteristic Curve':
            self.__log_roc_curve(figure, *curve_data)
        elif title == 'Partial Receiver Operating Characteristic Curve':
            self.__log_roc_curve(figure, *curve_data)
        else:
            raise NotImplementedError()

    def log_artifacts(self, models) -> None:
        # Log a model name, hyperparameters and metrics of all trained models into MLFlow
        num_models = len(models)
        for index, (_, model) in zip(range(num_models), models.iterrows()):
            mlflow.log_dict(dict(model), f'{index:0>{len(str(num_models))}}.json')

    def __log_pr_curve(self, figure, precision, recall, thresholds) -> None:
        mlflow.log_figure(figure, 'graph_pr_curve.png')
        mlflow.log_dict(
            {'precision': list(precision), 'recall': list(recall), 'thresholds': list(thresholds)},
            'data_pr_curve.json'
        )

    def __log_roc_curve(self, figure, fpr, tpr, thresholds) -> None:
        curve_name = 'roc_curve' if max(fpr) == 1 else 'partial_roc_auc'

        mlflow.log_figure(figure, f'graph_{curve_name}.png')
        mlflow.log_dict(
            {'fpr': list(fpr), 'tpr': list(tpr), 'thresholds': list(thresholds)},
            f'data_{curve_name}.json'
        )

    def set_tags(self, tags: dict[str, Any]) -> None:
        mlflow.set_tags(tags)


def log_duration(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        preprocessing_start = timer()
        ret = func(*args, **kwargs)
        preprocessing_end = timer()

        mlflow.log_metric('preprocessing_time', preprocessing_end - preprocessing_start)

        return ret
    return wrapper
