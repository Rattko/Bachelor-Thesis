from imblearn.over_sampling import ADASYN

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class AdasynResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1.0],
        'n_neighbors': [5, 10]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float = 1.0,
        n_neighbors: int = 5, random_state: int = None
    ) -> None:
        super().__init__(logger)

        self.resampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
