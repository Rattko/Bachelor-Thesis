""" Module implementing a unified representation of datasets used in experiments. """

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """ Class providing a unified way of working with datasets.

    Attributes
    ----------
    dataset_id : int | str
        Integer ID to identify the dataset uniquely.
    name : str
        Name of the dataset.
    size : int
        Number of samples contained in the dataset.
    majority_size : int
        Number of samples in the dataset belonging to the majority class.
    minority_size : int
        Number of samples in the dataset belonging to the minority class.
    imbalance : float
        Real number between 0 and 1 describing the degree of imbalance within the dataset.
    url : Optional[str]
        Link to an online datasets repository such as OpenML or Kaggle.
    data : Optional[np.ndarray]
        Matrix containing data of shape (n_samples, n_features).
    target : Optional[np.ndarray]
        Vector containing target values of shape (n_samples,).
    """

    dataset_id: int | str
    name: str
    size: int
    majority_size: int
    minority_size: int
    imbalance: float
    url: str = None
    data: np.ndarray = field(default=None, repr=False, compare=False)
    target: np.ndarray = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.size = int(self.size)
        self.majority_size = int(self.majority_size)
        self.minority_size = int(self.minority_size)

    def to_dict(self) -> dict[str, Any]:
        """ Convert dataclass fields to a dictionary.

        A field ends up in a dictionary only if it has the `repr` attribute set to True.

        Returns
        -------
        fields : dict[str, Any]
            Dictionary containing field names as keys and field values as values.
        """

        return {fld.name: getattr(self, fld.name) for fld in fields(self) if fld.repr}
