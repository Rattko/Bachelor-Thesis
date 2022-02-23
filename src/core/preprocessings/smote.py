import numpy as np
from imblearn.over_sampling import SMOTE

from core.preprocessings.resampler import Resampler

class SmoteResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'k_neighbors': [5, 10]
    }

    def __init__(self, sampling_strategy: float, k_neighbors: int, random_state: int) -> None:
        self.resampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=-1
        )

    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.resampler.fit_resample(data, target)
