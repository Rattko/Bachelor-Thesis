from typing import Any

from imblearn.under_sampling import NeighbourhoodCleaningRule

from core.preprocessings.resampler import Resampler

class NeighbourhoodCleaningRuleResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': ['not minority'],
        'n_neighbors': [5, 10],
        'kind_sel': ['all', 'mode'],
        'threshold_cleaning': [0.5, 0.75]
    }

    def __init__(
        self, sampling_strategy: str, n_neighbors: int,
        kind_sel: str, threshold_cleaning: float, **kwargs: Any
    ) -> None:
        super().__init__()

        self.resampler = NeighbourhoodCleaningRule(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            kind_sel=kind_sel,
            threshold_cleaning=threshold_cleaning,
            n_jobs=-1
        )
