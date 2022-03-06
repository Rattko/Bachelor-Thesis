from imblearn.over_sampling import BorderlineSMOTE

from core.preprocessings.resampler import Resampler

class BorderlineSmoteResampler(Resampler):
    _hyperparams = {
        'sampling_strategy': [0.75, 1],
        'k_neighbors': [5, 10],
        'm_neighbors': [5, 10],
        'kind': ['borderline-1', 'borderline-2']
    }

    def __init__(
        self, sampling_strategy: float, k_neighbors: int,
        m_neighbors: int, kind: str, random_state: int
    ) -> None:
        super().__init__()

        self.resampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            m_neighbors=m_neighbors,
            kind=kind,
            random_state=random_state,
            n_jobs=-1
        )
