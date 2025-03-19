from typing import Any, Optional, Tuple

from torch.nn import Module
from torch.utils.data import Dataset
from datasets import get_dataset
from datasets.abs_data import AbstractData
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
    """

    def __init__(self, args) -> None:
        self.args = args
        self._data = get_dataset(args.dataset, args.dataset_path, args=args)

        self._model = get_model(
            args.model, num_classes=10, factor=1, dims_feature=args.dims_feature
        )

    def get_data(self) -> AbstractData:
        return self._data

    def get_data_split(self) -> Tuple[Dataset, Dataset, Dataset]:
        auth_data = self._data.get_authentic_data()
        syn_data = self._data.get_synthetic_data()
        test_data = self._data.get_test_data()
        return (auth_data, syn_data, test_data)

    def get_model(self) -> Module:
        return self._model

    def get_output(self, id: Optional[int] = None) -> str:
        output = self.args.output if self.args.output else self.generate_name()
        if id is not None:
            return f"{output}_{id}.pt"
        return output + ".pt"

    def partition_data(self) -> Any:
        return self._data.partition_data()

    def generate_name(self) -> str:
        """
        Returns a name that will contains most important information of the
        experiment, also the arguments should not contain any invalid character
        for a file
        """
        name = "-".join(
            [
                f"E_{self.args.E}",
                self.args.model,
                self.args.dataset,
                f"eps_{self.args.epsilon:.2f}".replace(".", "_"),
                f"rho_{self.args.rho_syn / self.args.rho_tot:.2f}".replace(".", "_"),
                f"syn_balance_{self.args.syn_balance}",
                f"auth_balance_{1 if self.args.auth_balance else 0}",
                f"init_syn_{self.args.init_syn_rounds}",
            ]
        )
        return name


def get_wrapper() -> Wrapper:
    return Wrapper(ExperimentArgument())
