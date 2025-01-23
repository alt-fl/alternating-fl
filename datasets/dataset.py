from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset

from .cifar10 import CIFAR10Data
from .cinic10 import CINIC10Data


def get_dataset(name: str, path: Path) -> Tuple[Dataset, Dataset, Dataset]:
    """
    name: the dataset name, used to determine which dataset to use
    path: the path to the dataset
    """
    match name.lower():
        case "cifar10":
            data = CIFAR10Data(name, path)
        case "cinic10":
            data = CINIC10Data(name, path)
        case _:
            raise ValueError(f"Dataset {name!r} is not supported!")

    auth_set = data.get_authentic_data()
    syn_set = data.get_synthetic_data()
    test_set = data.get_test_data()
    return (auth_set, syn_set, test_set)
