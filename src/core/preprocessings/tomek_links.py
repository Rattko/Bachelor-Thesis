import numpy as np
from imblearn.under_sampling import TomekLinks

from core.preprocessings.resampler import Resampler

class TomekLinksResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority']
    }

    def __init__(self, sampling_strategy: str, random_state: int) -> None:
        self.resampler = TomekLinks(
            sampling_strategy=sampling_strategy,
            n_jobs=-1
        )

    def fit_resample(self, data: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.resampler.fit_resample(data, target)
