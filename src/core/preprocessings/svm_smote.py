""" Module wrapping SVMSMOTE from Imbalanced Learn. """

from imblearn.over_sampling import SVMSMOTE

from core.logger import Logger
from core.preprocessings.resampler import Resampler


class SvmSmoteResampler(Resampler):
    """ Wrapper over SVMSMOTE from Imbalanced Learn.

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
        'k_neighbors': [5, 10],
        'm_neighbors': [5, 10]
    }

    def __init__(
        self, logger: Logger, sampling_strategy: float = 1.0, k_neighbors: int = 5,
        m_neighbors: int = 10, random_state: int = None
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
