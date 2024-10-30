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
        poly_modulus_degree=2**13,
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

        self.context = get_context()
        self.enc_params = {}
        s = time.time()
        self.mask = get_enc_mask(self.args, self.nn, self.dataset, self.dict_users)
        e = time.time()
        print(f"Calculating encryption mask took {e - s}s")

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
        acc_list = []
        similarity_dict = {"feature": [], "classifier": []}
        if fedbn:
            acc_list_dict = {
                "MNIST": [],
                "SVHN": [],
                "USPS": [],
                "SynthDigits": [],
                "MNIST-M": [],
            }
            datasets_name = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST-M"]
        else:
            acc_list = []

        results_path = f"results/{self.args.dataset}/plain-fedfa/client_{int(self.args.C * self.args.K)}_{self.args.K}/"
        client_times = []
        gpu_utilizations = []
        memory_usages = []
        message_sizes = []

        for t in range(self.args.r):
            print("Round", t + 1, ":")
            # sampling
            np.random.seed(self.args.seed + t)
            m = np.max([int(self.args.C * self.args.K), 1])  # C is client sample rate
            index = np.random.choice(
                range(0, self.args.K), m, replace=False
            )  # sample m clients
            self.index_dict[t] = index

            # dispatch
            if fedbn:
                for i in index:
                    global_w = self.nn.state_dict()
                    client_w = self.nns[i].state_dict()
                    for key in global_w:
                        if "bn" not in key:
                            client_w[key] = global_w[key]
                    self.nns[i].load_state_dict(client_w)
            else:
                dispatch(index, self.nn, self.nns)
            dispatch(index, self.anchorloss, self.cls)

            # joint updating to obtain personalzied model based on updating global model
            start_time = time.time()
            tracemalloc.start()
            self.cls, self.nns, self.loss_dict, enc_params_dict = client_fedfa_cl(
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
            )
            end_time = time.time()

            # monitor different statistics
            current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
            memory_usages.append((peak_mem_usage, current_mem_usage))
            tracemalloc.stop()

            client_times.append(end_time - start_time)
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

            # compute feature similarity
            if similarity:
                # compute feature similarity
                mean_feature_similarity = compute_mean_feature_similarity(
                    self.args,
                    index,
                    self.nns,
                    self.dataset,
                    self.dict_users,
                    testset,
                    dict_users_test,
                )

                # compute classifier similarity
                client_classifiers = {i: [] for i in index}
                cos_sim_matrix = torch.zeros(len(index), len(index))
                for k in index:
                    classifier_weight_update = (
                        self.nns[k].classifier.weight.data
                        - self.nn.classifier.weight.data
                    )
                    classifier_bias_update = self.nns[k].classifier.bias.data.view(
                        10, 1
                    ) - self.nn.classifier.bias.data.view(10, 1)
                    client_classifiers[k] = torch.cat(
                        [classifier_weight_update, classifier_bias_update], 1
                    )
                for p, k in enumerate(index):
                    for q, j in enumerate(index):
                        cos_sim = torch.cosine_similarity(
                            client_classifiers[k], client_classifiers[j]
                        )
                        # print(cos_sim)
                        cos_sim_matrix[p][q] = torch.mean(cos_sim)
                mean_classifiers_similarity = torch.mean(cos_sim_matrix)

                similarity_dict["feature"].append(mean_feature_similarity)
                similarity_dict["classifier"].append(mean_classifiers_similarity)

            # aggregation
            if fedbn:
                aggregation(index, self.nn, self.nns, self.dict_users, fedbn=True)
            else:
                aggregation(index, self.nn, self.nns, self.dict_users)
                msg_size = asizeof.asizeof(self.nns)
                message_sizes.append(msg_size)
            aggregation(index, self.anchorloss, self.cls, self.dict_users)
            self.enc_params = fhe_aggregate(index, enc_params_dict, self.dict_users)

            if test_global_model_accuracy:
                if fedbn:
                    for index1, testset_per in enumerate(testset):
                        acc, _ = test_on_globaldataset_mixed_digit(
                            self.args,
                            self.nn,
                            testset_per,
                            dict_users_test[datasets_name[index1]],
                        )
                        acc_list_dict[datasets_name[index1]].append(acc)
                        print(acc)

                else:
                    # test accuracy on encrypted model --> we expect really bad accuracy
                    enc_acc, _ = test_on_globaldataset(self.args, self.nn, testset)

                    dec_model = copy.deepcopy(self.nn)
                    for name, param in dec_model.named_parameters():
                        if name not in self.mask:
                            continue
                        dec_params = self.enc_params[name].decrypt()
                        param_flat = param.data.view(-1)
                        param_flat[self.mask[name]] = torch.tensor(dec_params).to(
                            self.args.device
                        )

                    # test accuracy on decrypted model --> should see better accuracy
                    real_acc, _ = test_on_globaldataset(self.args, dec_model, testset)
                    acc_list.append((enc_acc, real_acc))

                    print(f"acc (encrypted): {enc_acc.item():.2f}%")
                    print(f"acc (decrypted): {real_acc.item():.2f}%")

            if (t + 1) % 10 == 0:
                print(
                    f"Checkpoint of data for dataset {self.args.dataset} made at round {t + 1}."
                )
                torch.save(
                    {
                        "client_times": client_times,
                        "gpu_utilizations": gpu_utilizations,
                        "memory_usages": memory_usages,
                        "message_sizes": message_sizes,
                    },
                    results_path + f"{self.args.dataset}_plain_resources.ckpt",
                )
                torch.save(
                    {
                        "model": self.nn.state_dict(),
                        "acc_list": acc_list,
                        "loss_dict": self.loss_dict,
                    },
                    results_path + f"{self.args.dataset}_plain_model.ckpt",
                )

        if fedbn:
            mean_CKA_dict = acc_list_dict
        else:
            mean_CKA_dict = acc_list

        # for k in range(self.args.K):
        #     path = f"results/cifar10/plain-fedfa/client_{int(self.args.C * self.args.K)}_{self.args.K}/"
        #     path += f"client{k}_model.pt"
        #     if self.nns[k] != []:
        #         torch.save(self.nns[k].state_dict(), path)

        torch.save(
            {
                "client_times": client_times,
                "gpu_utilizations": gpu_utilizations,
                "memory_usages": memory_usages,
                "message_sizes": message_sizes,
            },
            results_path + f"{self.args.dataset}_plain_resources.ckpt",
        )
        print("Saving resource allocation statistics...")
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


def compute_mean_feature_similarity(
    args, index, client_models, trainset, dict_users_train, testset, dict_users_test
):
    pdist = nn.PairwiseDistance(p=2)
    dict_class_verify = {i: [] for i in range(args.num_classes)}
    for i in dict_users_test:
        for c in range(args.num_classes):
            if np.array(testset.targets)[i] == c:
                dict_class_verify[c].append(i)
    # dict_clients_features = {k: {i: [] for i in range(args.num_classes)} for k in range(args.K)}
    dict_clients_features = {k: [] for k in index}
    for k in index:
        # labels = np.array(trainset.targets)[list(dict_users_train[k])]
        # labels_class = set(labels.tolist())
        # for c in labels_class:
        for c in range(args.num_classes):
            features_oneclass = verify_feature_consistency(
                args, client_models[k], testset, dict_class_verify[c]
            )
            features_oneclass = features_oneclass.view(
                1, features_oneclass.size()[0], features_oneclass.size()[1]
            )
            if c == 0:
                dict_clients_features[k] = features_oneclass
            else:
                dict_clients_features[k] = torch.cat(
                    [dict_clients_features[k], features_oneclass]
                )

    cos_sim_matrix = torch.zeros(len(index), len(index))
    for p, k in enumerate(index):
        for q, j in enumerate(index):
            for c in range(args.num_classes):
                cos_sim0 = pdist(
                    dict_clients_features[k][c], dict_clients_features[j][c]
                )
                # cos_sim0 = torch.cosine_similarity(dict_clients_features[k][c],
                #                                   dict_clients_features[j][c])
                # cos_sim0 = get_cos_similarity_postive_pairs(dict_clients_features[k][c],
                #                                    dict_clients_features[j][c])
                if c == 0:
                    cos_sim = cos_sim0
                else:
                    cos_sim = torch.cat([cos_sim, cos_sim0])
            cos_sim_matrix[p][q] = torch.mean(cos_sim)
    mean_feature_similarity = torch.mean(cos_sim_matrix)

    return mean_feature_similarity


def get_cos_similarity_postive_pairs(target, behaviored):
    attention_distribution_mean = []
    for j in range(target.size(0)):
        attention_distribution = []
        for i in range(behaviored.size(0)):
            attention_score = torch.cosine_similarity(
                target[j], behaviored[i].view(1, -1)
            )
            attention_distribution.append(attention_score)
        attention_distribution = torch.Tensor(attention_distribution)
        mean = torch.mean(attention_distribution)
        attention_distribution_mean.append(mean)
    attention_distribution_mean = torch.Tensor(attention_distribution_mean)
    return attention_distribution_mean
