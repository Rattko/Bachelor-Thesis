from imblearn.over_sampling import SVMSMOTE

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class SvmSmoteResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'k_neighbors': [5, 10],
        'm_neighbors': [5, 10]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float,
        k_neighbors: int, m_neighbors: int, random_state: int
    ) -> None:
        super().__init__(logger)

        self.resampler = SVMSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
            random_state=random_state,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
