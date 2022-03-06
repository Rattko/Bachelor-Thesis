from typing import Any

from imblearn.under_sampling import NearMiss

from core.preprocessings.resampler import Resampler

class NearMissResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'n_neighbors': [5, 10],
        'version': [1, 2, 3]
    }

    def __init__(
        self, sampling_strategy: float, n_neighbors: int, version: int, **kwargs: Any
    ) -> None:
        super().__init__()

        self.resampler = NearMiss(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            version=version,
            n_jobs=-1
        ) if version != 3 else NearMiss(
            sampling_strategy=sampling_strategy,
            n_neighbors_ver3=n_neighbors,
            version=version,
            n_jobs=-1
        )
