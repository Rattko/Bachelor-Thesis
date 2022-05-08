""" Module wrapping EditedNearestNeighbours from Imbalanced Learn. """

from typing import Any

from imblearn.under_sampling import EditedNearestNeighbours

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class EditedNearestNeighboursResampler(Resampler):
    """ Wrapper over EditedNearestNeighbours from Imbalanced Learn.

    Attributes
    ----------
    _hyperparams : dict[str, Any]
        Dictionary containing names and possible values of hyperparameters of the resampler.

    See Also
    --------
    The documentation and description for this and many more methods [1].

    [1]: https://imbalanced-learn.org/stable/index.html
    """

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

        self.resampler = EditedNearestNeighbours(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            kind_sel=kind_sel,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
