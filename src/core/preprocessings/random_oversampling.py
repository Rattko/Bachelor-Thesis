from imblearn.over_sampling import RandomOverSampler

from core.preprocessings.resampler import Resampler

class RandomOversamplingResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
    }

    def __init__(self, sampling_strategy: float, random_state: int) -> None:
        super().__init__()

        self.resampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
