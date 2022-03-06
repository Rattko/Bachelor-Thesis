from imblearn.over_sampling import KMeansSMOTE

from core.preprocessings.resampler import Resampler

class KMeansSmoteResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'k_neighbors': [5, 10]
    }

    def __init__(self, sampling_strategy: float, k_neighbors: int, random_state: int) -> None:
        super().__init__()

        self.resampler = KMeansSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=-1
        )
