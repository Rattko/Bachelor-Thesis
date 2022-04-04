from typing import Any

from imblearn.under_sampling import AllKNN

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class AllKnnResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority'],
        'n_neighbors': [5, 10],
        'kind_sel': ['all', 'mode']
    }

    def __init__(
        self, logger: Logger, sampling_strategy: str = 'not minority',
        n_neighbors: int = 3, kind_sel: str = 'all', **kwargs: Any
    ) -> None:
        super().__init__(logger)

        self.resampler = AllKNN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            kind_sel=kind_sel,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
