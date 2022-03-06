from typing import Any

from imblearn.under_sampling import TomekLinks

from core.preprocessings.resampler import Resampler

class TomekLinksResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority']
    }

    def __init__(self, sampling_strategy: str, **kwargs: Any) -> None:
        super().__init__()

        self.resampler = TomekLinks(
            sampling_strategy=sampling_strategy,
            n_jobs=-1
        )
