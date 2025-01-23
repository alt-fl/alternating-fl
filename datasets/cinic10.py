from pathlib import Path
from typing import Any
from torch.utils.data import Dataset

from .cifar10 import CIFAR10Data


class CINIC10Data(CIFAR10Data):
    def __init__(self, name: str, path: Path, args: Any = None, **kwargs) -> None:
        super().__init__(name, path, args, **kwargs)
