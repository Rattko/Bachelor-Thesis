from imblearn.under_sampling import ClusterCentroids

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class ClusterCentroidsResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'voting': ['hard', 'soft']
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float, voting: str, random_state: int
    ) -> None:
        super().__init__(logger)

        self.resampler = ClusterCentroids(
            sampling_strategy=sampling_strategy,
            voting=voting,
            random_state=random_state
        )

        self._logger.log_params('imblearn', self.get_params())
