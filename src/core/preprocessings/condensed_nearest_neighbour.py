from imblearn.under_sampling import CondensedNearestNeighbour

from core.preprocessings.resampler import Resampler

class CondensedNearestNeighbourResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority'],
        'n_neighbors': [5, 10],
        'n_seeds_S': [1]
    }

    def __init__(
        self, sampling_strategy: str, n_neighbors: int, n_seeds_S: int, random_state: int
    ) -> None:
        super().__init__()

        self.resampler = CondensedNearestNeighbour(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            n_seeds_S=n_seeds_S,
            random_state=random_state,
            n_jobs=-1
        )
