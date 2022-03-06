from typing import Any

import numpy as np

from core.preprocessings.resampler import Resampler

class BaselineResampler(Resampler):
    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return data, target

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}
