import time
from copy import deepcopy
from pathlib import Path
from typing import Optional
import tracemalloc
from pympler.asizeof import asizeof


import torch.cuda as cuda
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import tenseal as ts

from client import Client
from datasets.abs_data import AbstractData
from datasets.syn import InterleavingRounds
from exp_args import ExperimentArgument
from he.enc_mask import get_enc_mask
from tracker import Tracker
from training.epochs import EpochTransition, NoTransition
from training.fedfa.aggregate import fedavg_aggregate, he_aggregate
from training.fedfa.AnchorLoss import AnchorLoss

from logger import logger


class Server:
    def __init__(
        self,
        model: nn.Module,
        data: AbstractData,
        dataset: Dataset,
        client_data_idxs: dict,
        syn_dst: Dataset,
        syn_client_data_idxs: dict,
        output_path: Path,
        client_context: Optional[bytes] = None,
        server_context: Optional[bytes] = None,
        epoch_transition: EpochTransition = NoTransition(),
    ):
        self.args = ExperimentArgument()
        self.output_path = output_path

        self.tracker = Tracker(
            basic=[
                "is_round_authentic",
                "round_time",
                "gpu_util",
                "memory_usage",
                "model_size",
                "anchorloss_size",
                "ciphertext_size",
                "training_res",
                "fedfa_aggregation_time",
                "he_aggregate_time",
                "accuracy",
                "loss",
                "model",
                "anchorloss",
            ],
            limited=[("encryption_mask", 1), ("arguments", 1)],
            output_path=output_path,
        )
        self.tracker.track("arguments", deepcopy(self.args))

        self.clients: dict[int, Client] = {}
        self.global_model = model
        self.global_anchorloss = None
        if self.args.strategy.lower() == "fedfa":
            self.global_anchorloss = AnchorLoss(
                self.args.num_classes, self.args.dims_feature
            )

        # if homomorphic encryption is enabled, then we need to calculate
        # the mask and dissenminate it to the clients
        self.mask = None
        self.enc_params = None
        if self.args.epsilon > 0:
            # server gets a copy of the context but without the secret key
            self.context = ts.context_from(server_context)

            mask_time = time.time()
            self.mask = get_enc_mask(
                self.args,
                self.global_model,
                dataset,
                client_data_idxs,
                ratio=self.args.epsilon,
            )
            mask_time = time.time() - mask_time
            self.tracker.track("encryption_mask", deepcopy(self.mask))
            logger.info(f"Calculating encryption mask took {mask_time:.2f}s")

        # initialize the clients
        for id in range(self.args.K):
            self.clients[id] = Client(
                id,
                self.args,
                deepcopy(self.global_model),
                dataset,
                client_data_idxs[id],
                anchorloss=deepcopy(self.global_anchorloss),
                syn_dataset=syn_dst,
                syn_data_indices=syn_client_data_idxs[id],
                he_context=client_context,
                mask=self.mask,
                epoch_func=epoch_transition,
            )

        self.data = data
        self.client_data_idxs = client_data_idxs
        self.syn_client_data_idxs = syn_client_data_idxs

    def start_training(self):
        num_activate_clients = max(int(self.args.C * self.args.K), 1)
        # begin the communication rounds
        for round_num, is_auth in InterleavingRounds(
            self.args.r,
            ratio=(self.args.rho_syn, self.args.rho_tot),
            syn_only=self.args.init_syn_rounds,
        ):
            auth_str = "Authentic" if is_auth else "Synthetic"
            logger.info(f"=== {auth_str} Round {round_num + 1} ===")

            self.tracker.track("is_round_authentic", is_auth)

            client_idxs = np.random.choice(
                np.array(list(self.clients.keys())), num_activate_clients, replace=False
            )

            data_idxs = self.client_data_idxs if is_auth else self.syn_client_data_idxs

            client_models = {}
            client_anchors = {}
            client_enc_params = {}
            client_enc_models = {}
            client_training_res = {}

            round_time = time.time()
            # sequentially train all the clients
            for id in client_idxs:
                # each rounds begin by broadcasting the global models and
                # anchorloss to the clients
                model_copy = deepcopy(self.global_model)
                anchorloss_copy = deepcopy(self.global_anchorloss)
                self.clients[id].dispatch(model_copy, anchorloss_copy)

                res = self.clients[id].train(
                    round_num, is_auth_round=is_auth, enc_params=self.enc_params
                )
                client_models[id] = res["model"]
                client_anchors[id] = res["anchorloss"]
                client_training_res[id] = res["training_res"]

                if "enc_params" in res and "enc_model" in res:
                    client_enc_params[id] = res["enc_params"]
                    client_enc_models[id] = res["enc_model"]
            round_time = time.time() - round_time

            _, peak_mem_usage = tracemalloc.get_traced_memory()
            self.tracker.track("round_time", round_time)
            self.tracker.track("memory_usage", peak_mem_usage)
            self.tracker.track("training_res", client_training_res)
            if cuda.is_available():
                gpu_util = (
                    cuda.utilization(),
                    (cuda.max_memory_allocated(), cuda.memory_allocated()),
                )
                self.tracker.track("gpu_util", gpu_util)

            real_client_models = client_models
            if client_enc_models:
                # the real model depends on if we are using encrypted models this round
                real_client_models = client_enc_models

            agg_time = time.time()
            # aggregate clients models to the global model
            agg_model = fedavg_aggregate(client_idxs, real_client_models, data_idxs)
            self.global_model.load_state_dict(agg_model)

            if self.args.strategy.lower() == "fedfa":
                # aggregate client anchors if we are using FedFA
                agg_anchor = fedavg_aggregate(client_idxs, client_anchors, data_idxs)
                self.global_anchorloss.load_state_dict(agg_anchor)  # type: ignore

            agg_time = time.time() - agg_time
            model_size = asizeof(self.global_model)
            anchorloss_size = asizeof(self.global_anchorloss)
            self.tracker.track("fedfa_aggregation_time", agg_time)
            self.tracker.track("model_size", model_size)
            self.tracker.track("anchorloss_size", anchorloss_size)

            # reset enc_params previous round
            self.enc_params = None
            he_agg_time = 0
            ciphertext_size = 0
            # aggregate the encrypted parameters for each client if needed
            if client_enc_params:
                he_agg_time = time.time()
                self.enc_params = he_aggregate(
                    client_idxs, client_enc_params, data_idxs
                )
                he_agg_time = time.time() - he_agg_time
                ciphertext_size = asizeof(self.enc_params)
            self.tracker.track("ciphertext_size", ciphertext_size)
            self.tracker.track("he_aggregate_time", he_agg_time)

            test_model = deepcopy(self.global_model)
            if client_enc_models:
                # we always test on the unencrypted model since encrypted model
                # performance doesn't matter
                unenc_agg_model = fedavg_aggregate(
                    client_idxs, client_models, data_idxs
                )
                test_model.load_state_dict(unenc_agg_model)

            acc, loss = self.data.test(test_model)
            self.tracker.track("accuracy", acc)
            self.tracker.track("loss", loss)
            logger.info(
                f"Round {round_num + 1}: accuracy = {acc:.2%}, loss = {loss:.3f}"
            )
            logger.info(f"Round {round_num + 1} completed in {round_time:.2f}s\n")

            if (round_num + 1) % self.args.save_every == 0:

                self.tracker.track("model", deepcopy(self.global_model.state_dict()))
                self.tracker.track(
                    "anchorloss",
                    (
                        deepcopy(self.global_anchorloss.state_dict())
                        if self.global_anchorloss
                        else None
                    ),
                )
                self.tracker.save()
                logger.info(
                    f"Checkpoint saved at {str(self.output_path)}, size {self.output_path.stat().st_size / 1e6 :.2f}MB\n"
                )
