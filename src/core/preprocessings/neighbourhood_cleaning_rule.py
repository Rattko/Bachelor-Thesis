from typing import Any

from imblearn.under_sampling import NeighbourhoodCleaningRule

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class NeighbourhoodCleaningRuleResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority'],
        'n_neighbors': [5, 10],
        'kind_sel': ['all', 'mode'],
        'threshold_cleaning': [0.5, 0.75]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: str = 'not minority', n_neighbors: int = 3,
        kind_sel: str = 'all', threshold_cleaning: float = 0.5, **kwargs: Any
    ) -> None:
        super().__init__(logger)

        self.resampler = NeighbourhoodCleaningRule(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            kind_sel=kind_sel,
            threshold_cleaning=threshold_cleaning,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
