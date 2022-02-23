import itertools
from typing import Any, Generator

import numpy as np

class Resampler:
    _hyperparams = {}

    @classmethod
    def hyperparams(cls) -> Generator[dict[str, Any], None, None]:
        keys = cls._hyperparams.keys()

        for conf in itertools.product(*cls._hyperparams.values()):
            yield dict(zip(keys, conf))

    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
