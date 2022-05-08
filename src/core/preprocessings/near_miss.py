""" Module wrapping NearMiss from Imbalanced Learn. """

from typing import Any

from imblearn.under_sampling import NearMiss

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class NearMissResampler(Resampler):
    """ Wrapper over NearMiss from Imbalanced Learn.

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
        'sampling_strategy': [0.75, 1.0],
        'n_neighbors': [5, 10],
        'version': [1, 2, 3]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float = 1.0,
        n_neighbors: int = 3, version: int = 1, **kwargs: Any
    ) -> None:
        super().__init__(logger)

        self.resampler = NearMiss(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            version=version,
            n_jobs=-1
        ) if version != 3 else NearMiss(
            sampling_strategy=sampling_strategy,
            n_neighbors_ver3=n_neighbors,
            version=version,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
