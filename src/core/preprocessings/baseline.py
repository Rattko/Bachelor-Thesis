import numpy as np

from core.preprocessings.resampler import Resampler

class BaselineResampler(Resampler):
    def __init__(self, random_state: int) -> None:
        pass

    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return data, target
