from typing import Tuple

from torch.nn import Module
from torch.utils.data import Dataset
from datasets import get_dataset
from exp_args import ExperimentArgument
from models import get_model


class Wrapper:
    """
    Wrapper class for handling the specified experiment according to the
    arguments

    It's mainly responsible for handling the following components in the
    experiments, namely:
        1. authentic and synthetic datasets
        2. instantiating the specified model
        3. handling interleaving ratio
        4. dynamic epoch, interleaving ratio
    """

    def __init__(self, args) -> None:
        self.data = get_dataset(args.dataset, args.dataset_path)

        self.model = get_model(
            args.model, num_classes=10, factor=1, dims_feature=args.dims_feature
        )

    def get_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        return self.data

    def get_model(self) -> Module:
        return self.model


def get_wrapper() -> Wrapper:
    return Wrapper(ExperimentArgument())
