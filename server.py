import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np
import tenseal as ts

import time
import tracemalloc

from pympler import asizeof


import model
from client import *
from utils.aggregator import *
from utils.dispatchor import *
from utils.optimizer import *
from utils.global_test import *
from utils.local_test import *
from utils.sampling import *
from utils.AnchorLoss import *
from utils.ContrastiveLoss import *
from utils.CKA import linear_CKA, kernel_CKA

from exp_args import ExperimentArgument

from datasets import InterleavingRounds
from he import get_he_context, get_enc_mask
from training.epochs import get_transition


def seed_torch(seed, test=True):
    if test:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Server:

    def __init__(
        self, model, dataset, dict_users, syn_dst, syn_dict_users, output_path
    ):
        self.args = ExperimentArgument()
        seed_torch(self.args.seed)
        self.output = output_path

        self.nn = copy.deepcopy(model)
        self.nns = [[] for i in range(self.args.K)]
        self.p_nns = []
        self.cls = []
        self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict = dict((k, [0]) for k in key)
        # self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict = dict((i, []) for i in range(self.args.r))
        self.dataset = dataset
        self.dict_users = dict_users

        self.syn_dst = syn_dst
        self.syn_dict_users = syn_dict_users

        self.epoch_transition = get_transition(self.args)

        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(
            self.args.device
        )
        for i in range(self.args.K):
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2)

        self.enc_params = {}
        self.context = None
        self.sk = None
        self.mask = None

        print(f"\ninterleaving ratio: {self.args.rho_syn / self.args.rho_tot:.2f}")

        if self.args.epsilon > 0:
            print(f"selective homomorphic encryption ratio: {self.args.epsilon}")
            self.context = get_he_context()
            self.sk = self.context.secret_key()
            s = time.time()
            self.mask = get_enc_mask(
                self.args,
                self.nn,
                self.dataset,
                self.dict_users,
                ratio=self.args.epsilon,
            )
            e = time.time()
            print(f"calculating encryption mask took {e - s:.2f}s")

    def fedfa_anchorloss(
        self,
        testset,
        dict_users_test,
        test_global_model_accuracy=False,
    ):
        checkpoint_path = self.output
        print(f"the statistics will be logged to {checkpoint_path!r}")
        similarity_dict = {"feature": [], "classifier": []}

        round_types = []
        acc_list = []
        round_times = []
        gpu_utilizations = []
        memory_usages = []
        message_sizes = []
        anchorloss_sizes = []
        training_times = []
        ciphertext_sizes = []
        agg_times = []
        fhe_agg_times = []

        last_models = {}

        for t, is_auth in InterleavingRounds(
            rounds=self.args.r,
            ratio=(self.args.rho_syn, self.args.rho_tot),
            syn_only=self.args.init_syn_rounds,
        ):
            print(f"\n{'Authentic' if is_auth else 'Synthetic'} round {t + 1}:")
            # sampling
            np.random.seed(self.args.seed + t)
            m = np.max([int(self.args.C * self.args.K), 1])  # C is client sample rate
            index = np.random.choice(
                range(0, self.args.K), m, replace=False
            )  # sample m clients
            self.index_dict[t] = index

            # use synthetic data if sythentic round
            if is_auth:
                dst = self.dataset
                dict_users = self.dict_users
            else:
                dst = self.syn_dst
                dict_users = self.syn_dict_users
            round_types.append(is_auth)

            # dispatch
            dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)

            he_context = self.context.serialize() if self.context else None
            # joint updating to obtain personalzied model based on updating global model
            start_time = time.time()

            self.cls, self.nns, self.loss_dict, training_stats, enc_params_dict = (
                client_fedfa_cl(
                    self.args,
                    index,
                    self.cls,
                    self.nns,
                    self.nn,
                    t,
                    self.epoch_transition,
                    dst,
                    dict_users,
                    self.loss_dict,
                    he_context,
                    self.sk,
                    self.mask,
                    self.enc_params,
                    is_auth,
                )
            )
            end_time = time.time()

            # monitor different statistics
            _, peak_mem_usage = tracemalloc.get_traced_memory()
            round_times.append(end_time - start_time)
            training_times.append(training_stats)

            if torch.cuda.is_available():
                gpu_utilizations.append(
                    (
                        torch.cuda.utilization(),
                        (
                            torch.cuda.max_memory_allocated(),
                            torch.cuda.memory_allocated(),
                        ),
                    )
                )

            # aggregate the model
            start_time = time.time()
            aggregation(index, self.nn, self.nns, self.dict_users)
            end_time = time.time()
            fedavg_agg_time = end_time - start_time

            msg_size = asizeof.asizeof(self.nns)
            message_sizes.append(msg_size)

            # aggregate feature anchors
            start_time = time.time()
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            end_time = time.time()
            anchor_agg_time = end_time - start_time
            agg_times.append((fedavg_agg_time, anchor_agg_time))
            print(f"FedFA aggregation time: {fedavg_agg_time + anchor_agg_time:.2f}s")

            anchorloss_size = asizeof.asizeof(self.anchorloss)
            anchorloss_sizes.append(anchorloss_size)

            if self.args.epsilon > 0 and enc_params_dict:
                start_time = time.time()
                self.enc_params = fhe_aggregate(
                    index,
                    enc_params_dict,
                    self.dict_users,
                )
                end_time = time.time()
                fhe_agg_time = end_time - start_time
                fhe_agg_times.append(fhe_agg_time)
                print(f"FHE aggregation time: {fhe_agg_time:.2f}s")

                ciphertext_size = asizeof.asizeof(list(self.enc_params.values()))
                ciphertext_sizes.append(ciphertext_size)
            else:
                fhe_agg_times.append(0)
                ciphertext_sizes.append(0)
                self.enc_params = {}

            if test_global_model_accuracy:
                test_start = time.time()
                # test accuracy on encrypted model --> we expect really bad accuracy
                enc_acc, _ = test_on_globaldataset(self.args, self.nn, testset)

                if self.args.epsilon > 0 and self.enc_params:
                    dec_model = copy.deepcopy(self.nn)
                    for name, param in dec_model.named_parameters():
                        if name not in self.mask:
                            continue
                        dec_params = self.enc_params[name].decrypt(self.sk)
                        with torch.no_grad():
                            param_flat = param.view(-1)
                            param_flat[self.mask[name]] = torch.tensor(dec_params).to(
                                self.args.device
                            )

                    # test accuracy on decrypted model --> should see better accuracy
                    real_acc, _ = test_on_globaldataset(self.args, dec_model, testset)
                    acc_list.append(real_acc)
                    global_acc = real_acc

                    print(f"acc (encrypted): {enc_acc.item():.2f}%")
                    print(f"acc (decrypted): {real_acc.item():.2f}%")
                else:
                    # if no encryption, then the model is not encrypted and
                    # enc_acc gives us the real test accuracy
                    acc_list.append(enc_acc)
                    global_acc = enc_acc
                    print(f"acc: {enc_acc.item():.2f}%")
                test_end = time.time()
                test_time = test_end - test_start
                print(f"testing time: {test_time:.2f}s")

            if (t + 1) % self.args.save_every == 0:
                last_models[t] = {
                    "model": copy.deepcopy(self.nn.state_dict()),
                    "anchorloss": copy.deepcopy(self.anchorloss.state_dict()),
                }
                torch.save(
                    {
                        "round_types": round_types,
                        "round_times": round_times,
                        "gpu_utilizations": gpu_utilizations,
                        "memory_usages": memory_usages,
                        "message_sizes": message_sizes,
                        "anchorloss_sizes": anchorloss_sizes,
                        "training_times": training_times,
                        "ciphertext_sizes": ciphertext_sizes,
                        "agg_times": agg_times,
                        "fhe_agg_times": fhe_agg_times,
                        "models": last_models,
                        "acc_list": acc_list,
                        "loss_dict": self.loss_dict,
                        "mask": copy.deepcopy(self.mask),
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint of {self.args.model} made at round {t + 1}")

            # don't track the recorded statistics in the memory
            stats_size = sum(
                asizeof.asizesof(
                    round_types,
                    acc_list,
                    round_times,
                    gpu_utilizations,
                    memory_usages,
                    message_sizes,
                    anchorloss_sizes,
                    training_times,
                    ciphertext_sizes,
                    agg_times,
                    fhe_agg_times,
                    last_models,
                )
            )
            memory_usages.append(peak_mem_usage - stats_size)
            print(f"peak memory usage: {(peak_mem_usage - stats_size) / 1e6:.2f}MB")

        mean_CKA_dict = acc_list

        torch.save(
            {
                "round_types": round_types,
                "round_times": round_times,
                "gpu_utilizations": gpu_utilizations,
                "memory_usages": memory_usages,
                "message_sizes": message_sizes,
                "anchorloss_sizes": anchorloss_sizes,
                "training_times": training_times,
                "ciphertext_sizes": ciphertext_sizes,
                "agg_times": agg_times,
                "fhe_agg_times": fhe_agg_times,
                "models": last_models,
                "acc_list": acc_list,
                "loss_dict": self.loss_dict,
                "mask": copy.deepcopy(self.mask),
            },
            checkpoint_path,
        )

        self.nns = [[] for i in range(self.args.K)]
        torch.cuda.empty_cache()
        return (
            self.nn,
            similarity_dict,
            self.nns,
            self.loss_dict,
            self.index_dict,
            mean_CKA_dict,
        )
