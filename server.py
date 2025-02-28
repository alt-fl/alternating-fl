import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import tenseal as ts

from client import Client
from datasets.abs_data import AbstractData
from datasets.syn import InterleavingRounds
from exp_args import ExperimentArgument
from he.enc_mask import get_enc_mask
from training.epochs import EpochTransition, NoTransition
from training.fedfa.aggregate import fedavg_aggregate, he_aggregate
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
        client_context: Optional[bytes] = None,
        server_context: Optional[bytes] = None,
        epoch_transition: EpochTransition = NoTransition(),
    ):
        self.args = ExperimentArgument()

        self.clients: dict[int, Client] = {}
        self.global_model = model
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
            logger.info(f"Calculating encryption mask took {mask_time:.2f}s")

        # initialize the clients
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

            client_idxs = np.random.choice(
                np.array(list(self.clients.keys())), num_activate_clients, replace=False
            )

            data_idxs = self.client_data_idxs if is_auth else self.syn_client_data_idxs

            start_time = time.time()
            # sequentially train all the clients
            client_models = {}
            client_anchors = {}
            client_enc_params = {}
            client_enc_models = {}
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
                if "enc_params" in res and "enc_model" in res:
                    client_enc_params[id] = res["enc_params"]
                    client_enc_models[id] = res["enc_model"]
            end_time = time.time()

            real_client_models = client_models
            if client_enc_models:
                # the real model depends on if we are using encrypted models this round
                real_client_models = client_enc_models

            # aggregate clients models to the global model
            agg_model = fedavg_aggregate(client_idxs, real_client_models, data_idxs)
            self.global_model.load_state_dict(agg_model)

            # aggregate client anchors
            agg_anchor = fedavg_aggregate(client_idxs, client_anchors, data_idxs)
            self.global_anchorloss.load_state_dict(agg_anchor)

            # reset enc_params previous round
            self.enc_params = None
            # aggregate the encrypted parameters for each client if needed
            if client_enc_params:
                self.enc_params = he_aggregate(
                    client_idxs, client_enc_params, data_idxs
                )

            test_model = deepcopy(self.global_model)
            if client_enc_models:
                # we always test on the unencrypted model since encrypted model
                # performance doesn't matter
                unenc_agg_model = fedavg_aggregate(
                    client_idxs, client_models, data_idxs
                )
                test_model.load_state_dict(unenc_agg_model)

            acc, loss = self.data.test(test_model)
            logger.info(
                f"Round {round_num + 1}: accuracy = {acc:.2%}, loss = {loss:.3f}"
            )
            logger.info(
                f"Round {round_num + 1} completed in {end_time - start_time:.2f}s\n"
            )
