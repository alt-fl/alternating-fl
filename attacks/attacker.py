from abc import ABC, abstractmethod
from copy import deepcopy

from torch.nn import Module
from torch.utils.data import Dataset

from training.fedavg.optim import optimize
from exp_args import ExperimentArgument


class Attacker(ABC):
    @abstractmethod
    def reconstruct(self, data_idx: int | list[int], **kwargs) -> dict:
        """
        Launch the attack on the image specified by its index, or a batch of
        images specified by the list of indices.
        Returns a dictionary containing attack results.
        """
        pass

    def simulate(
        self, model: Module, dataset: Dataset, data_idx: int | list[int], **kwargs
    ):
        """
        Simulates training of a model on the specified data.
        Returns the resulting gradients.
        """
        if isinstance(data_idx, int):
            data_idx = [data_idx]

        # yep, we can just use the global argument parser
        args = ExperimentArgument()

        tmp_model = deepcopy(model)  # we don't want to modify the model
        # simulate a single epoch of update
        optimize(args, tmp_model, dataset, data_idx, num_epoch=1, use_dp=False)
        grads = [
            param.grad.detach().clone()
            for param in tmp_model.parameters()
            if param.grad is not None
        ]
        return grads
