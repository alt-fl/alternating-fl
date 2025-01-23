from abc import ABC, abstractmethod
from pathlib import Path
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
