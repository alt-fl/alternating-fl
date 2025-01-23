from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from .abs_data import AbstractData
from .syn import partition_data


class CIFAR10Data(AbstractData):
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path

        # precalculated CIFAR10 mean and std
        mean = [0.491, 0.482, 0.447]
        std = [0.247, 0.243, 0.262]
        transf = Compose([ToTensor(), Normalize(mean=mean, std=std)])

        train_data = CIFAR10(
            root=self.path, train=True, download=True, transform=transf
        )

        self.auth_data, self.syn_data = partition_data(train_data, train_data.classes)
        self.test_data = CIFAR10(
            root=self.path, train=False, download=True, transform=transf
        )

    def get_authentic_data(self) -> Dataset:
        return self.auth_data

    def get_synthetic_data(self) -> Dataset:
        return self.syn_data

    def get_test_data(self) -> Dataset:
        return self.test_data
