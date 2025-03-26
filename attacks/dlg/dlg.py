"""
Implements the DLG and iDLG attacks from the papers
https://arxiv.org/abs/1906.08935 and https://arxiv.org/abs/2001.02610

Code is adapted from
https://github.com/PatrickZH/Improved-Deep-Leakage-from-Gradients
"""

import torch

from torch.nn import Module, CrossEntropyLoss
from torch.optim.lbfgs import LBFGS
from torch.utils.data import Dataset

from copy import deepcopy
from typing import Optional

from sewar.full_ref import msssim, vifp, uqi

from attacks.attacker import Attacker
from attacks.utils import weights_init
from logger import logger


class DLGAttacker(Attacker):
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        num_classes: int,
        weights: Optional[dict] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Parameters:
            model: FL model used during training.
            dataset: dataset from which the image is retrieved.
            weights: state_dict of the model, if not provided then will random
                     initialized weights
            device: device on which to run computations.
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.num_classes = num_classes
        self.device = device

        self.weights = weights

    def reconstruct(
        self, data_idx: int | list[int], lr=1.0, num_iters=300, use_idlg=False, **kwargs
    ) -> dict:
        # tt = ToTensor()
        # tp = ToPILImage()
        model = deepcopy(self.model)

        model.apply(weights_init)
        if self.weights:
            model.load_state_dict(self.weights)

        if isinstance(data_idx, int):
            # turn the specified image to be attack to a list for convenience
            data_idx = [data_idx]

        model = model.to(self.device)
        num_dummy = len(data_idx)

        criterion = CrossEntropyLoss().to(self.device)
        real_data = real_label = None

        # starting the reconstruction here
        for i in range(num_dummy):
            idx = data_idx[i]
            # tmp_datum = tt(self.dataset[idx][0]).float().to(self.device)
            # the dataset is already transformed
            tmp_datum = self.dataset[idx][0].float().to(self.device)
            tmp_datum = tmp_datum.view(1, *tmp_datum.size())

            tmp_label = torch.Tensor([self.dataset[idx][1]]).long().to(self.device)
            tmp_label = tmp_label.view((1,))
            if (real_data is None) or (real_label is None):
                # if we are attacking the first image
                real_data = tmp_datum
                real_label = tmp_label
            else:
                real_data = torch.cat((real_data, tmp_datum), dim=0)
                real_label = torch.cat((real_label, tmp_label), dim=0)

        if (real_data is None) or (real_label is None):
            raise ValueError("no image to be reconstructed")

        # compute original gradient (this is the original step from DLG/iDLG)
        # _, out = model(real_data)
        # y = criterion(out, real_label)
        # dy_dx = torch.autograd.grad(y, model.parameters())  # type: ignore
        # original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        # simulate original gradient (we skip it by simulating an actual step)
        original_dy_dx = self.simulate(model, self.dataset, data_idx, **kwargs)

        # generate dummy data and label
        dummy_data = torch.randn(real_data.size()).to(self.device).requires_grad_(True)
        dummy_label = (
            torch.randn((real_data.shape[0], self.num_classes))
            .to(self.device)
            .requires_grad_(True)
        )

        if use_idlg:
            optimizer = LBFGS([dummy_data], lr=lr)
            # predict the ground-truth label
            label_pred = (
                torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1)
                .detach()
                .reshape((1,))
                .requires_grad_(False)
            )
        else:
            optimizer = LBFGS([dummy_data, dummy_label], lr=lr)
            label_pred = torch.tensor(-1)  # just to prevent linter complaining

        losses = []
        mses = []
        for iter in range(num_iters):

            def closure():
                optimizer.zero_grad()
                _, pred = model(dummy_data)

                if use_idlg:
                    dummy_loss = criterion(pred, label_pred)
                else:
                    dummy_loss = -torch.mean(
                        torch.sum(
                            torch.softmax(dummy_label, -1)
                            * torch.log(torch.softmax(pred, -1)),
                            dim=-1,
                        )
                    )

                dummy_dy_dx = torch.autograd.grad(
                    dummy_loss, model.parameters(), create_graph=True
                )

                grad_diff = torch.tensor(0.0)
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            current_loss = closure().item()
            losses.append(current_loss)

            mse = torch.mean((dummy_data - real_data) ** 2)
            mses.append(mse.item())

            if (iter + 1) % 10 == 0:
                logger.debug(f"Loss: {current_loss:.2f}, MSE: {mse:.2f}")

        if use_idlg:
            pred_label = label_pred.item()
        else:
            pred_label = torch.argmax(dummy_label, dim=-1).detach().item()

        return {
            "loss": losses,
            "label_guess": pred_label,
            "label_real": real_label,
            "mse": mses,
        }
