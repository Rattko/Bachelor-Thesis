from imblearn.over_sampling import SMOTE

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class SmoteResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1.0],
        'k_neighbors': [5, 10]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float = 1.0,
        k_neighbors: int = 5, random_state: int = None
    ) -> None:
        super().__init__(logger)

        self.resampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=random_state,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
