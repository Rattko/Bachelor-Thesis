from typing import Any

from imblearn.under_sampling import EditedNearestNeighbours

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class EditedNearestNeighboursResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority'],
        'n_neighbors': [5, 10],
        'kind_sel': ['all', 'mode']
    }

    def __init__(
        self, logger: Logger, sampling_strategy: str,
        n_neighbors: int, kind_sel: str, **kwargs: Any
    ) -> None:
        super().__init__(logger)

        self.resampler = EditedNearestNeighbours(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            kind_sel=kind_sel,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
