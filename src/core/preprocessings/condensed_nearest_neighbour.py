from imblearn.under_sampling import CondensedNearestNeighbour

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class CondensedNearestNeighbourResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority'],
        'n_neighbors': [5, 10],
        'n_seeds_S': [1]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: str = 'not minority',
        n_neighbors: int = 5, n_seeds_S: int = 1, random_state: int = None
    ) -> None:
        super().__init__(logger)

        self.resampler = CondensedNearestNeighbour(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            n_seeds_S=n_seeds_S,
            random_state=random_state,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
