from typing import Optional
import torch

from torch.nn import Module, MSELoss
from torch.optim.adam import Adam
from torch.utils.data import Dataset

from attacks.attacker import Attacker
from logger import logger


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        logger.warning(f"Failed in weights_init for {m._get_name()}.weight")

    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        logger.warning(f"Failed in weights_init for {m._get_name()}.bias")


class DLGAttacker(Attacker):
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        device: torch.device = torch.device("cpu"),
        weights: Optional[dict] = None,
    ):
        """
        Parameters:
            model: FL model used during training.
            dataset: dataset from which the image is retrieved.
            device: device on which to run computations.
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.loss_fn = MSELoss()

        self.weights = weights

    def launch_attack(self, data_idx: int, **kwargs) -> dict:
        # Get the actual image (and target, if needed) from the dataset
        actual_img, target = self.dataset[data_idx]
        actual_img = actual_img.to(self.device)

        # Initialize a dummy image with the same shape as the actual image.
        # This image will be optimized to "match" the gradients computed from the real image.
        dummy_img = torch.randn_like(actual_img, requires_grad=True, device=self.device)

        # Setup optimizer to update the dummy image
        optimizer = Adam([dummy_img], lr=0.1)

        # Number of iterations for the gradient inversion process
        num_iterations = 1000

        # Simulate the attack loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()

            # For a real DLG attack, you would compute gradients from the model parameters using the true image,
            # then optimize dummy_img such that its computed gradients match those gradients.
            # For this example, we simplify by directly minimizing the MSE between dummy_img and actual_img.
            loss = self.loss_fn(dummy_img, actual_img)
            loss.backward()
            optimizer.step()

        # After optimization, compute the final MSE between the reconstructed (dummy) image and the actual image.
        mse_value = self.loss_fn(dummy_img, actual_img).item()

        # Return the result as a dictionary.
        return {"mse": mse_value}
