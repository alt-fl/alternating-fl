from pathlib import Path
from typing import Any, Tuple
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from fedlab.utils.dataset.partition import CIFAR10Partitioner
from utils.sampling import trainset_sampling_label

from .abs_data import AbstractData
from .syn import balance_auth_dst, partition_data


class CIFAR10Data(AbstractData):
    def __init__(self, name: str, path: Path, args: Any = None, **kwargs) -> None:
        self.name = name
        self.path = path
        self.args = args

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

    def partition_data(self) -> Tuple[Any, Any]:
        if not self.args.auth_balance:
            auth_part = CIFAR10Partitioner(
                self.auth_data.targets,
                num_clients=self.args.K,
                balance=True,
                partition="iid",
                seed=1,
            )
        else:
            auth_part = CIFAR10Partitioner(
                self.auth_data.targets,
                num_clients=self.args.K,
                balance=None,
                partition="dirichlet",
                dir_alpha=0.5,
                seed=1,
            )

        # dict users sampling from original FedFA code, not a very elegant
        # solution but it will have to do...
        rare_class_number = 0  # i don't know why this is 0
        auth_dict_users = trainset_sampling_label(
            self.args,
            self.auth_data,
            # the sample rate is not touched in any experiment, always 1
            self.args.trainset_sample_rate,
            rare_class_number,
            auth_part,
        )

        match self.args.syn_balance.lower():
            case "self":
                syn_part = CIFAR10Partitioner(
                    self.syn_data.targets,
                    num_clients=self.args.K,
                    balance=True,
                    partition="iid",
                    seed=1,
                )
                syn_dict_users = trainset_sampling_label(
                    self.args,
                    self.syn_data,
                    self.args.trainset_sample_rate,
                    rare_class_number,
                    syn_part,
                )
            case "all":
                syn_dict_users = balance_auth_dst(
                    self.auth_data,
                    auth_dict_users,
                    self.syn_data,
                    self.args.num_classes,
                )

            case _:
                raise ValueError(
                    f"syn_balance does not accept value {self.args.syn_balance!r}"
                )

        return (auth_dict_users, syn_dict_users)
