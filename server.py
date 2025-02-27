import time
from copy import deepcopy
from pathlib import Path

import torch.nn as nn
from torch.utils.data import Dataset

from client import Client
from datasets.abs_data import AbstractData
from datasets.syn import InterleavingRounds
from exp_args import ExperimentArgument
from training.fedfa.aggregate import fedavg_aggregate
from utils.AnchorLoss import AnchorLoss

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
    ):
        self.args = ExperimentArgument()

        self.clients = {}
        self.global_model = model
        self.global_anchorloss = AnchorLoss(
            self.args.num_classes, self.args.dims_feature
        )
        for id in range(self.args.K):
            self.clients[id] = Client(
                id,
                self.args,
                deepcopy(self.global_model),
                deepcopy(self.global_anchorloss),
                dataset,
                client_data_idxs[id],
                syn_dataset=syn_dst,
                syn_data_indices=syn_client_data_idxs[id],
            )

        self.data = data
        self.client_data_idxs = client_data_idxs
        self.syn_client_data_idxs = syn_client_data_idxs

    def start_training(self):
        # begin the communication rounds
        for round_num, is_auth in InterleavingRounds(
            self.args.r,
            ratio=(self.args.rho_syn, self.args.rho_tot),
            syn_only=self.args.init_syn_rounds,
        ):
            auth_str = "Authentic" if is_auth else "Synthetic"
            logger.info(f"=== {auth_str} Round {round_num + 1} ===")

            for client in self.clients.values():
                # each rounds begin by broadcasting the global models and
                # anchorloss to the clients
                model_copy = deepcopy(self.global_model)
                anchorloss_copy = deepcopy(self.global_anchorloss)
                client.dispatch(model_copy, anchorloss_copy)

            start_time = time.time()

            # sequentially train all the clients
            client_models = {}
            client_anchors = {}
            for client in self.clients.values():
                res = client.train()
                id = res["id"]
                client_models[id] = res["model"]
                client_anchors[id] = res["anchorloss"]

            end_time = time.time()

            # aggregate updates and update global model + anchors
            agg_model = fedavg_aggregate(
                self.clients.keys(), client_models, self.client_data_idxs
            )
            self.global_model.load_state_dict(agg_model)

            agg_anchor = fedavg_aggregate(
                self.clients.keys(), client_anchors, self.client_data_idxs
            )
            self.global_anchorloss.load_state_dict(agg_anchor)

            acc, loss = self.data.test(self.global_model)
            logger.info(
                f"Round {round_num + 1} completed in {end_time - start_time:.2f}s"
            )
            logger.info(
                f"Round {round_num + 1}: accuracy = {acc * 100:.2f}%, loss = {loss:.3f}\n"
            )
