from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple
from torch.utils.data import Dataset


class AbstractData(ABC):
    """
    Parent class that abstracts how we preprocess the specified dataset
    """

    @abstractmethod
    def __init__(self, name: str, path: Path) -> None:
        pass

    @abstractmethod
    def get_authentic_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_synthetic_data(self) -> Dataset:
        pass

    @abstractmethod
    def get_test_data(self) -> Dataset:
        pass

    @abstractmethod
    def partition_data(self) -> Tuple[Any, Any]:
        """
        Return a tuple of partitioner class from fedlab, one for authentic,
        and one for synthetic data
        """
        pass
