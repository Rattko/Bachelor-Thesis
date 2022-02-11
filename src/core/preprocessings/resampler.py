import numpy as np

class Resampler:
    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
