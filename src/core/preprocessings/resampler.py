import itertools
from typing import Any, Generator

import numpy as np
from imblearn.base import BaseSampler

from core.logger import Logger


class Resampler:
    _hyperparams: dict[str, Any] = {}

    def __init__(self, logger: Logger, **kwargs: Any) -> None:
        self.resampler: BaseSampler = None
        self._logger = logger

    @classmethod
    def hyperparams(cls) -> Generator[dict[str, Any], None, None]:
        keys = cls._hyperparams.keys()

        for conf in itertools.product(*cls._hyperparams.values()):
            yield dict(zip(keys, conf))

    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.resampler.fit_resample(data, target)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return self.resampler.get_params(deep)
