import time

from argparse import Namespace
from copy import deepcopy
from typing import Any, Iterable, Optional

import psutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import tenseal as ts

from training.epochs import EpochTransition, NoTransition
from training.fedfa.optim import optimize
from training.fedfa.AnchorLoss import AnchorLoss

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
    ) -> None:
        self.args = args

        # fedfa
        self.id = id
        self.model = model
        self.dataset = dataset
        self.data_indices = data_indices
        self.anchorloss = anchorloss

        # alt-fl
        self.syn_dataset = syn_dataset
        self.syn_data_indices = syn_data_indices

        # homomorphic encryption
        self.context = self.skey = self.mask = None
        if self.args.epsilon > 0:
            # NOTE that in practice the context should be securely exchanged between
            # the clients, and not from the server, but in a controlled environment,
            # e.g. the server never actually uses the context, it is acceptable
            # for simulation purpose
            self.context = ts.context_from(he_context) if he_context else None
            self.skey = self.context.secret_key() if self.context else None
            self.mask = mask
            assert self.context and self.skey and self.mask

        # dynamic local epoch
        self.epoch_func = epoch_func

    def train(
        self,
        round_num: int,
        is_auth_round: bool = True,
        enc_params: Optional[dict[str, ts.CKKSVector]] = None,
    ) -> dict[str, Any]:
        """
        Train the client for a communication round, where the number of local
        training epochs are determined by self.epoch_func.

        By default, the communication round is always authentic, unless an
        interleaving ratio is specified.

        If encrypted parameters are given and epsilon > 0, then the params will
        be decrypted and assigned to the model accordingly. Similarly, if
        epsilon > 0 and the round is authentic, then the model will be selectively
        encrypted before sending to the server.

        Differential privacy is controlled by the optimize function.
        """
        num_epoch = self.epoch_func.estimate_epoch(round_num)
        training_res = {}
        training_res["num_epoch"] = num_epoch

        # kickstart CPU utilization tracing
        psutil.cpu_percent()

        # HE decryption
        training_res["decryption"] = 0
        if enc_params and self.args.epsilon > 0:
            dec_time = time.time()
            self.decrypt_and_update(enc_params)
            dec_time = time.time() - dec_time

            training_res["decryption"] = dec_time
            logger.debug(f"Client {self.id} decryption time {dec_time:.2f}s")

        logger.debug(f"Client {self.id} training start")
        if self.args.epoch_transition and round_num < self.args.transition_rounds:
            # print out the estimated number epoch as long as it's still during
            # the transition rounds
            logger.debug(f"Client {self.id} local training epochs = {num_epoch}")

        training_time = time.time()
        loss = optimize(
            self.args,
            self.anchorloss,
            self.model,
            self.dataset if is_auth_round else self.syn_dataset,
            self.data_indices if is_auth_round else self.syn_data_indices,
            num_epoch=num_epoch,
            comm_round=round_num,
            # synthetic rounds don't need DP, just like HE
            use_dp=self.args.use_dp and is_auth_round,
        )
        training_time = time.time() - training_time
        training_loss = sum(loss) / len(loss)
        training_res["training"] = training_time
        logger.debug(f"Client {self.id} training loss: {training_loss:.2f}")
        logger.info(f"Client {self.id} training time: {training_time:.2f}s")

        training_res["cpu_util"] = psutil.cpu_percent()

        # NOTE that the model is actually not encrypted here
        res = {
            "id": self.id,
            "model": deepcopy(self.model.state_dict()),
            "anchorloss": deepcopy(self.anchorloss.state_dict()),
            "loss": training_loss,
        }
        # encryption
        if self.args.epsilon > 0 and is_auth_round:
            enc_time = time.time()
            res["enc_params"] = self.encrypt_and_update()
            # we don't override the actual model because we need that for
            # evaluation purposes
            res["enc_model"] = deepcopy(self.model.state_dict())
            enc_time = time.time() - enc_time
            training_res["encryption"] = enc_time
            logger.info(f"Client {self.id} encryption time: {enc_time:.2f}s")

        res["training_res"] = training_res
        return res

    def decrypt_and_update(self, enc_params: dict[str, ts.CKKSVector]) -> None:
        """
        Given the encrypted parameters, decrypts them and updates the current
        model with it
        """
        # mostly just to shut the linter up
        assert self.mask is not None and self.context is not None

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.mask:
                    continue

                dec_param = enc_params[name].decrypt(self.skey)
                param_flat = param.view(-1)
                param_flat[self.mask[name]] = torch.tensor(dec_param).to(
                    self.args.device
                )

    def encrypt_and_update(self) -> dict[str, ts.CKKSVector]:
        """
        Encrypts the parameters in current according to the mask, and updatesthe model by hiding encrypted parameters
        """
        # mostly just to shut the linter up
        assert self.mask is not None and self.context is not None

        enc_params: dict[str, ts.CKKSVector] = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.mask:
                    continue

                param_flat = param.view(-1)
                enc_param_flat = ts.ckks_vector(
                    self.context, param_flat[self.mask[name]].cpu().detach().clone()
                )

                enc_params[name] = enc_param_flat
                # hide encrypted parameters by setting random values
                param_flat[self.mask[name]] = torch.randn(self.mask[name].shape).to(
                    self.args.device
                )
        return enc_params

    def dispatch(self, global_model: nn.Module, anchorloss: nn.Module) -> None:
        self.model = global_model
        self.anchorloss = anchorloss
