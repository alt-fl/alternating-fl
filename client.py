import time

from argparse import Namespace
from copy import deepcopy
from typing import Any, Iterable, Optional

import torch.nn as nn
from torch.utils.data import Dataset

import tenseal as ts

from training.epochs import EpochTransition, NoTransition
from training.fedfa.optim import optimize
from utils.AnchorLoss import AnchorLoss

from logger import logger


class Client:
    def __init__(
        self,
        id: int,
        args: Namespace,
        model: nn.Module,
        anchorloss: AnchorLoss,
        dataset: Dataset,
        data_indices: Iterable[int],
        syn_dataset: Optional[Dataset] = None,
        syn_data_indices: Optional[Iterable[int]] = None,
        he_context: Optional[bytes] = None,
        mask: Optional[dict] = None,
        epoch_func: EpochTransition = NoTransition(),
        init_round: int = 0,
    ) -> None:
        self.args = args

        # fedfa
        self.id = id
        self.model = model
        self.dataset = dataset
        self.data_indices = data_indices
        self.anchorloss = anchorloss
        self.round = init_round

        # alt-fl
        self.syn_dataset = syn_dataset
        self.syn_data_indices = syn_data_indices

        # homomorphic encryption
        self.context = ts.context_from(he_context) if he_context else None
        self.skey = self.context.secret_key() if self.context else None
        self.mask = mask

        # dynamic local epoch
        self.epoch_func = epoch_func

    def train(self, is_auth_round: bool = True) -> dict[str, Any]:
        # train for a certain number of epochs for a communication round,
        # by default the communication round is always authentic

        num_epoch = self.epoch_func.estimate_epoch(self.round)
        start_time = time.time()

        logger.debug(f"Client {self.id} training start")
        loss = optimize(
            self.args,
            self.anchorloss,
            self.model,
            self.dataset if is_auth_round else self.syn_dataset,
            self.data_indices if is_auth_round else self.syn_data_indices,
            num_epoch=num_epoch,
            comm_round=self.round,
        )
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Client {self.id} training time: {training_time:.2f}")

        return {
            "id": self.id,
            "model": deepcopy(self.model.state_dict()),
            "anchorloss": deepcopy(self.anchorloss.state_dict()),
            "loss": [],
        }

    def dispatch(self, global_model: nn.Module, anchorloss: nn.Module) -> None:
        self.model = global_model
        self.anchorloss = anchorloss
