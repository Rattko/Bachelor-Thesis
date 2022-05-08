""" Module wrapping TomekLinks from Imbalanced Learn. """

from typing import Any

from imblearn.under_sampling import TomekLinks

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class TomekLinksResampler(Resampler):
    """ Wrapper over TomekLinks from Imbalanced Learn.

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
        'sampling_strategy': ['not minority']
    }

    def __init__(
        self, logger: Logger, sampling_strategy: str = 'not minority', **kwargs: Any
    ) -> None:
        super().__init__(logger)

        self.resampler = TomekLinks(
            sampling_strategy=sampling_strategy,
            n_jobs=-1
        )

        self._logger.log_params('imblearn', self.get_params())
