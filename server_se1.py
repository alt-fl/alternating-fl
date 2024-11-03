import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import copy
import random, os
import matplotlib.pyplot as plt
import numpy as np
import tenseal as ts


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
from enc_mask import get_enc_mask

import time
import tracemalloc

from pympler import asizeof


def get_context():
    # controls precision of the fractional part
    bits_scale = 26
    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            31,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            31,
        ],
    )
    # set the scale
    context.global_scale = pow(2, bits_scale)
    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()
    return context


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

    def __init__(self, args, model, dataset, dict_users):
        seed_torch(args.seed)
        self.args = args
        self.nn = copy.deepcopy(model)
        self.nns = [[] for i in range(self.args.K)]
        self.p_nns = []
        self.cls = []
        self.cocols = []
        self.contrals = []
        key = [i for i in range(self.args.K)]
        self.loss_dict = dict((k, [0]) for k in key)
        # self.finetune_loss_dict =  dict((k, [0]) for k in key)
        self.index_dict = dict((i, []) for i in range(args.r))
        self.dataset = dataset
        self.dict_users = dict_users

        self.anchorloss = AnchorLoss(self.args.num_classes, self.args.dims_feature).to(
            args.device
        )
        for i in range(self.args.K):
            temp2 = copy.deepcopy(self.anchorloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.cls.append(temp2)

        self.enc_params = {}
        self.context = None
        self.mask = None

        if self.args.ratio > 0:
            self.context = get_context()
            s = time.time()
            self.mask = get_enc_mask(
                self.args, self.nn, self.dataset, self.dict_users, ratio=self.args.ratio
            )
            e = time.time()
            print(f"Selective homomorphic encryption ratio: {self.args.ratio:.4f}")
            print(f"Calculating encryption mask took {e - s:.2f}s")

        self.contrastiveloss = ContrastiveLoss(
            self.args.num_classes, self.args.dims_feature
        ).to(args.device)
        for i in range(self.args.K):
            temp2 = copy.deepcopy(self.contrastiveloss)
            clients = [str(i) for i in range(self.args.K)]
            temp2.name = clients[i]
            self.contrals.append(temp2)

    def fedfa_anchorloss(
        self,
        testset,
        dict_users_test,
        similarity=False,
        fedbn=False,
        test_global_model_accuracy=False,
    ):
        similarity_dict = {"feature": [], "classifier": []}

        results_path = self.args.path
        acc_list = []
        round_times = []
        gpu_utilizations = []
        memory_usages = []
        message_sizes = []
        training_times = []
        ciphertext_sizes = []
        agg_times = []
        fhe_agg_times = []

        last_models = {}

        for t in range(self.args.r):
            print(f"\nRound {t + 1}:")
            # sampling
            np.random.seed(self.args.seed + t)
            m = np.max([int(self.args.C * self.args.K), 1])  # C is client sample rate
            index = np.random.choice(
                range(0, self.args.K), m, replace=False
            )  # sample m clients
            self.index_dict[t] = index

            # dispatch
            dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)

            # joint updating to obtain personalzied model based on updating global model
            start_time = time.time()
            tracemalloc.start()
            self.cls, self.nns, self.loss_dict, training_stats, enc_params_dict = (
                client_fedfa_cl(
                    self.args,
                    index,
                    self.cls,
                    self.nns,
                    self.nn,
                    t,
                    self.dataset,
                    self.dict_users,
                    self.loss_dict,
                    self.context,
                    self.mask,
                    self.enc_params,
                    self.context,
                )
            )
            end_time = time.time()

            # monitor different statistics
            current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
            memory_usages.append((peak_mem_usage, current_mem_usage))
            tracemalloc.stop()

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

            # # compute feature similarity
            # if similarity:
            #     # compute feature similarity
            #     mean_feature_similarity = compute_mean_feature_similarity(
            #         self.args,
            #         index,
            #         self.nns,
            #         self.dataset,
            #         self.dict_users,
            #         testset,
            #         dict_users_test,
            #     )
            #
            #     # compute classifier similarity
            #     client_classifiers = {i: [] for i in index}
            #     cos_sim_matrix = torch.zeros(len(index), len(index))
            #     for k in index:
            #         classifier_weight_update = (
            #             self.nns[k].classifier.weight.data
            #             - self.nn.classifier.weight.data
            #         )
            #         classifier_bias_update = self.nns[k].classifier.bias.data.view(
            #             10, 1
            #         ) - self.nn.classifier.bias.data.view(10, 1)
            #         client_classifiers[k] = torch.cat(
            #             [classifier_weight_update, classifier_bias_update], 1
            #         )
            #     for p, k in enumerate(index):
            #         for q, j in enumerate(index):
            #             cos_sim = torch.cosine_similarity(
            #                 client_classifiers[k], client_classifiers[j]
            #             )
            #             # print(cos_sim)
            #             cos_sim_matrix[p][q] = torch.mean(cos_sim)
            #     mean_classifiers_similarity = torch.mean(cos_sim_matrix)
            #
            #     similarity_dict["feature"].append(mean_feature_similarity)
            #     similarity_dict["classifier"].append(mean_classifiers_similarity)

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

            if self.args.ratio > 0:
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

            if test_global_model_accuracy:
                # test accuracy on encrypted model --> we expect really bad accuracy
                enc_acc, _ = test_on_globaldataset(self.args, self.nn, testset)

                if self.args.ratio > 0:
                    dec_model = copy.deepcopy(self.nn)
                    for name, param in dec_model.named_parameters():
                        if name not in self.mask:
                            continue
                        dec_params = ts.ckks_vector_from(
                            self.context, self.enc_params[name]
                        ).decrypt()
                        with torch.no_grad():
                            param_flat = param.view(-1)
                            param_flat[self.mask[name]] = torch.tensor(dec_params).to(
                                self.args.device
                            )

                    # test accuracy on decrypted model --> should see better accuracy
                    real_acc, _ = test_on_globaldataset(self.args, dec_model, testset)
                    acc_list.append((enc_acc, real_acc))

                    print(f"acc (encrypted): {enc_acc.item():.2f}%")
                    print(f"acc (decrypted): {real_acc.item():.2f}%")
                else:
                    # if no encryption, then the model is not encrypted and
                    # enc_acc gives us the real test accuracy
                    acc_list.append(enc_acc)
                    print(f"acc: {enc_acc.item():.2f}%")

            last_models[t + 1] = {
                "model": copy.deepcopy(self.nn.state_dict()),
                "anchorloss": copy.deepcopy(self.anchorloss.state_dict()),
                "enc_params": copy.deepcopy(self.enc_params),
            }

            # we keep the last 3 models
            for i in sorted(last_models.keys())[:-3]:
                del last_models[i]

            if (t + 1) % 2 == 0:
                torch.save(
                    {
                        "round_times": round_times,
                        "gpu_utilizations": gpu_utilizations,
                        "memory_usages": memory_usages,
                        "message_sizes": message_sizes,
                        "training_times": training_times,
                        "ciphertext_sizes": ciphertext_sizes,
                        "agg_times": agg_times,
                        "fhe_agg_times": fhe_agg_times,
                        "models": last_models,
                        "acc_list": acc_list,
                        "loss_dict": self.loss_dict,
                        "context": (
                            copy.deepcopy(self.context.serialize())
                            if self.args.ratio > 0
                            else None
                        ),
                    },
                    results_path + f"checkpoint{self.args.seed}.pt",
                )
                print(f"Checkpoint of {self.args.model} made at round {t + 1}")

        mean_CKA_dict = acc_list

        torch.save(
            {
                "round_times": round_times,
                "gpu_utilizations": gpu_utilizations,
                "memory_usages": memory_usages,
                "message_sizes": message_sizes,
                "training_times": training_times,
                "ciphertext_sizes": ciphertext_sizes,
                "agg_times": agg_times,
                "fhe_agg_times": fhe_agg_times,
                "models": last_models,
                "acc_list": acc_list,
                "loss_dict": self.loss_dict,
                "context": (
                    copy.deepcopy(self.context.serialize())
                    if self.args.ratio > 0
                    else None
                ),
            },
            results_path + f"checkpoint{self.args.seed}.pt",
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


#
# def compute_mean_feature_similarity(
#     args, index, client_models, trainset, dict_users_train, testset, dict_users_test
# ):
#     pdist = nn.PairwiseDistance(p=2)
#     dict_class_verify = {i: [] for i in range(args.num_classes)}
#     for i in dict_users_test:
#         for c in range(args.num_classes):
#             if np.array(testset.targets)[i] == c:
#                 dict_class_verify[c].append(i)
#     # dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}
#     dict_clients_features = {k: [] for k in index}
#     for k in index:
#         # labels = np.array(trainset.targets)[list(dict_users_train[k])]
#         # labels_class = set(labels.tolist())
#         # for c in labels_class:
#         for c in range(args.num_classes):
#             features_oneclass = verify_feature_consistency(
#                 args, client_models[k], testset, dict_class_verify[c]
#             )
#             features_oneclass = features_oneclass.view(
#                 1, features_oneclass.size()[0], features_oneclass.size()[1]
#             )
#             if c == 0:
#                 dict_clients_features[k] = features_oneclass
#             else:
#                 dict_clients_features[k] = torch.cat(
#                     [dict_clients_features[k], features_oneclass]
#                 )
#
#     cos_sim_matrix = torch.zeros(len(index), len(index))
#     for p, k in enumerate(index):
#         for q, j in enumerate(index):
#             for c in range(args.num_classes):
#                 cos_sim0 = pdist(
#                     dict_clients_features[k][c], dict_clients_features[j][c]
#                 )
#                 # cos_sim0 = torch.cosine_similarity(dict_clients_features[k][c],
#                 #                                   dict_clients_features[j][c])
#                 # cos_sim0 = get_cos_similarity_postive_pairs(dict_clients_features[k][c],
#                 #                                    dict_clients_features[j][c])
#                 if c == 0:
#                     cos_sim = cos_sim0
#                 else:
#                     cos_sim = torch.cat([cos_sim, cos_sim0])
#             cos_sim_matrix[p][q] = torch.mean(cos_sim)
#     mean_feature_similarity = torch.mean(cos_sim_matrix)
#
#     return mean_feature_similarity
#
#
# def get_cos_similarity_postive_pairs(target, behaviored):
#     attention_distribution_mean = []
#     for j in range(target.size(0)):
#         attention_distribution = []
#         for i in range(behaviored.size(0)):
#             attention_score = torch.cosine_similarity(
#                 target[j], behaviored[i].view(1, -1)
#             )
#             attention_distribution.append(attention_score)
#         attention_distribution = torch.Tensor(attention_distribution)
#         mean = torch.mean(attention_distribution)
#         attention_distribution_mean.append(mean)
#     attention_distribution_mean = torch.Tensor(attention_distribution_mean)
#     return attention_distribution_mean
