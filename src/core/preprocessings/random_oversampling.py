""" Module wrapping RandomOverSampler from Imbalanced Learn. """

from imblearn.over_sampling import RandomOverSampler

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class RandomOversamplingResampler(Resampler):
    """ Wrapper over RandomOverSampler from Imbalanced Learn.

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
        'sampling_strategy': [0.75, 1.0]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float = 1.0, random_state: int = None
    ) -> None:
        super().__init__(logger)

        self.resampler = RandomOverSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )

        self._logger.log_params('imblearn', self.get_params())
