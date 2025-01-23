from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset

from .cifar10 import CIFAR10Data
from .cinic10 import CINIC10Data
from .abs_data import AbstractData


def get_dataset(name: str, path: Path, **kwargs) -> AbstractData:
    """
    name: the dataset name, used to determine which dataset to use
    path: the path to the dataset
    """
    match name.lower():
        case "cifar10":
            return CIFAR10Data(name, path, **kwargs)
        case "cinic10":
            return CINIC10Data(name, path, **kwargs)
        case _:
            raise ValueError(f"Dataset {name!r} is not supported!")
