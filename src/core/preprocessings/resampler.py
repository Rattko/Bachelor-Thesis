""" Module implementing the base class for resampling methods. """

import itertools
from typing import Any, Generator

import numpy as np
from imblearn.base import BaseSampler

from core.logger import Logger, log_duration


class Resampler:
    """ Base class defining an interface for the resampling methods used in the framework.

    Attributes
    ----------
    _hyperparams : dict[str, Any]
        Dictionary containing names and possible values of hyperparameters of the resampler.
    """

    _hyperparams: dict[str, Any] = {}

    def __init__(self, logger: Logger, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        logger : Logger
            Instance of `Logger`.
        **kwargs : dict[str, Any]
            Hyperparameters used to instantiate the resampler.
        """

        self.resampler: BaseSampler = None
        self._logger = logger

    @classmethod
    def hyperparams(cls) -> Generator[dict[str, Any], None, None]:
        """ Generate all combinations of the resampler's hyperparameters.

        Yields
        ------
        hyperparams : dict[str, Any]
            Single configuration of hyperparameters of the resampler.
        """

        keys = cls._hyperparams.keys()

        for conf in itertools.product(*cls._hyperparams.values()):
            yield dict(zip(keys, conf))

    @log_duration
    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Fit the resampling method to the given data and resample the data.

        Parameters
        ----------
        data : np.ndarray
            Matrix containing training data of shape (n_samples, n_features).
        target : np.ndarray
            Vector containing target values of shape (n_samples,).

        Returns
        -------
        resampled_data : np.ndarray
            Resampled data matrix.
        resampled_target : np.ndarray
            Resampled target vector.
        """

        return self.resampler.fit_resample(data, target)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """ Get hyperparameters of the resampling method.

        Parameters
        ----------
        deep : bool
            If true, get hyperparameters of this resampler and any nested models.

        Returns
        -------
        params : dict[str, Any]
            Hyperparameter names mapped to their values.
        """

        return self.resampler.get_params(deep)
