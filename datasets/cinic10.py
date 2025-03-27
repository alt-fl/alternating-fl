from pathlib import Path
from typing import Any

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor

from .cifar10 import CIFAR10Data


class CINIC10Data(CIFAR10Data):
    def __init__(self, name: str, path: Path, args: Any = None, **kwargs) -> None:
        self.name = name
        self.path = path
        self.args = args

        # precalculated CINIC10 mean and std
        self.mean = [0.47889522, 0.47227842, 0.43047404]
        self.std = [0.24205776, 0.23828046, 0.25874835]
        transf = Compose([ToTensor(), Normalize(mean=self.mean, std=self.std)])

        cinic_train = ImageFolder(str(Path(path, "train/")), transform=transf)
        cinic_val = ImageFolder(str(Path(path, "valid/")), transform=transf)
        cinic_test = ImageFolder(str(Path(path, "test/")), transform=transf)

        self.auth_data = cinic_train
        self.syn_data = cinic_val
        self.test_data = cinic_test
