from imblearn.under_sampling import RandomUnderSampler

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class RandomUndersamplingResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1.0]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float = 1.0, random_state: int = None
    ) -> None:
        super().__init__(logger)

        self.resampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )

        self._logger.log_params('imblearn', self.get_params())
