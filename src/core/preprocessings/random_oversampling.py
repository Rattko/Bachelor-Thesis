from imblearn.over_sampling import RandomOverSampler

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class RandomOversamplingResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
    }

    def __init__(self, logger: Logger, sampling_strategy: float, random_state: int) -> None:
        super().__init__(logger)

        self.resampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )

        self._logger.log_params('imblearn', self.get_params())
