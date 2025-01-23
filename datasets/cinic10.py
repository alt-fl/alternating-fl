from pathlib import Path
from torch.utils.data import Dataset

from .abs_data import AbstractData


class CINIC10Data(AbstractData):
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path

    def get_authentic_data(self) -> Dataset:
        return super().get_authentic_data()

    def get_synthetic_data(self) -> Dataset:
        return super().get_synthetic_data()

    def get_test_data(self) -> Dataset:
        return super().get_test_data()
