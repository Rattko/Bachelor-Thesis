from imblearn.under_sampling import ClusterCentroids

from core.preprocessings.resampler import Resampler

class ClusterCentroidsResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'voting': ['hard', 'soft']
    }

    def __init__(self, sampling_strategy: float, voting: str, random_state: int) -> None:
        super().__init__()

        self.resampler = ClusterCentroids(
            sampling_strategy=sampling_strategy,
            voting=voting,
            random_state=random_state
        )
