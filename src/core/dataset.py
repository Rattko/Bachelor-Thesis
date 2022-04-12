from dataclasses import dataclass, field

import numpy as np

@dataclass
class Dataset:
    dataset_id: int
    name: str
    size: int
    majority_size: int
    minority_size: int
    imbalance: float
    url: str = None
    data: np.ndarray = field(default=None, repr=False, compare=False)
    target: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.size = int(self.size)
        self.majority_size = int(self.majority_size)
        self.minority_size = int(self.minority_size)
