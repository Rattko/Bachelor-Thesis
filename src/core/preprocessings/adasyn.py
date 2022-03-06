from imblearn.over_sampling import ADASYN

from core.preprocessings.resampler import Resampler

class AdasynResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'n_neighbors': [5, 10]
    }

    def __init__(self, sampling_strategy: float, n_neighbors: int, random_state: int) -> None:
        super().__init__()

        self.resampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=-1
        )
