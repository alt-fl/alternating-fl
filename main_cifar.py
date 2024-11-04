import torch
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import random, os, sys, re
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import pandas as pd
import copy

import time


from fedlab.utils.dataset import FMNISTPartitioner, CIFAR10Partitioner
from fedlab.utils.functional import partition_report, save_dict

from args_cifar10_c2 import args_parser
import server_se1 as server
import model

from utils.global_test import (
    test_on_globaldataset,
    globalmodel_test_on_localdataset,
    globalmodel_test_on_specifdataset,
)
from utils.local_test import test_on_localdataset
from utils.training_loss import train_loss_show, train_localacc_show
from utils.sampling import testset_sampling, trainset_sampling, trainset_sampling_label
from utils.tSNE import FeatureVisualize
from synthetic_data import SyntheticCIFAR10

args = args_parser()


class Filter(object):
    def __init__(self, stream, re_pattern):
        self.stream = stream
        self.pattern = (
            re.compile(re_pattern) if isinstance(re_pattern, str) else re_pattern
        )
        self.triggered = False

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == "\n" and self.triggered:
            self.triggered = False
        else:
            if self.pattern.search(data) is None:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()


# example
sys.stdout = Filter(
    sys.stdout, r"WARNING: The input does not fit in a single ciphertext"
)  # filter out any line which contains "Read -1" in it


def seed_torch(seed=args.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_FedFA():
    seed_torch()

    results_path = f"results/{args.model.lower()}/fhe{str(args.ratio).replace('.', '_')}_inter{args.AR}_{args.SR}/"
    args.path = results_path
    if not os.path.exists(results_path):
        print(f"Creating directory {results_path}")
        os.makedirs(results_path)
    else:
        print(f"Directory {results_path} already exists, proceeding...")

    similarity = False
    save_models = True
    Train_model = True

    C = "2CNN_2"
    specf_model = model.ClientModel(args, name="cifar10").to(args.device)
    trans_cifar10 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
        ]
    )
    root = "data/CIFAR10/"
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=trans_cifar10
    )
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=trans_cifar10
    )

    num_classes = args.num_classes
    num_clients = args.K
    number_perclass = args.num_perclass

    col_names = [f"class{i}" for i in range(num_classes)]
    print(col_names)
    # hist_color = "#4169E1"
    # plt.rcParams["figure.facecolor"] = "white"

    # perform partition
    noniid_labeldir_part = CIFAR10Partitioner(
        trainset.targets,
        num_clients=num_clients,
        balance=None,
        partition="shards",
        num_shards=200,
        seed=1,
    )
    # generate partition report
    csv_file = "data/CIFAR10/cifar10_noniid_labeldir_clients_10.csv"
    partition_report(
        trainset.targets,
        noniid_labeldir_part.client_dict,
        class_num=num_classes,
        verbose=False,
        file=csv_file,
    )

    noniid_labeldir_part_df = pd.read_csv(csv_file, header=1)
    noniid_labeldir_part_df = noniid_labeldir_part_df.set_index("client")
    for col in col_names:
        noniid_labeldir_part_df[col] = (
            noniid_labeldir_part_df[col] * noniid_labeldir_part_df["Amount"]
        ).astype(int)

    # select first 10 clients for bar plot
    # noniid_labeldir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    # # plt.tight_layout()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.xlabel("sample num")
    # plt.savefig(
    #     f"data/CIFAR10//cifar10_noniid_labeldir_clients_10.png",
    #     dpi=400,
    #     bbox_inches="tight",
    # )

    # split dataset into training and testing

    trainset_sample_rate = args.trainset_sample_rate

    rare_class_nums = 0
    dict_users_train = trainset_sampling_label(
        args, trainset, trainset_sample_rate, rare_class_nums, noniid_labeldir_part
    )
    dict_users_test = testset_sampling(
        args, testset, number_perclass, noniid_labeldir_part_df
    )

    # training_number = {j: {} for j in range(args.K)}
    #
    # for i in range(args.K):
    #     training_number[i] = {j: 0 for j in range(num_classes)}
    #     label_class = set(
    #         np.array(trainset.targets)[list(dict_users_train[i])].tolist()
    #     )
    #     # print(list(label_class))
    #     for k in label_class:
    #         training_number[i][k] = list(
    #             np.array(trainset.targets)[list(dict_users_train[i])]
    #         ).count(k)
    #
    # df_training_number = []
    # df_training_number = pd.DataFrame(df_training_number)
    # for i in range(args.K):
    #     temp = pd.Series(training_number[i])
    #     df_training_number[i] = temp
    #
    # df_training_number["Col_sum"] = df_training_number.apply(lambda x: x.sum(), axis=1)
    # df_training_number.loc["Row_sum"] = df_training_number.apply(lambda x: x.sum())
    #
    # test_number = {j: {} for j in range(args.K)}
    #
    # for i in range(args.K):
    #     test_number[i] = {j: 0 for j in range(num_classes)}
    #     label_class = set(np.array(testset.targets)[list(dict_users_test[i])].tolist())
    #     # print(list(label_class))
    #     for k in label_class:
    #         test_number[i][k] = list(
    #             np.array(testset.targets)[list(dict_users_test[i])]
    #         ).count(k)
    #
    # df_test_number = []
    # df_test_number = pd.DataFrame(df_test_number)
    # for i in range(args.K):
    #     temp = pd.Series(test_number[i])
    #     df_test_number[i] = temp
    #
    # df_test_number["Col_sum"] = df_test_number.apply(lambda x: x.sum(), axis=1)
    # df_test_number.loc["Row_sum"] = df_test_number.apply(lambda x: x.sum())
    #
    # # perform partition
    # iid_part = FMNISTPartitioner(
    #     trainset.targets, num_clients=num_clients, partition="iid", seed=1
    # )
    # # generate partition report
    # csv_file = "data/CIFAR10/cifar10_iid_clients_10.csv"
    # partition_report(
    #     trainset.targets,
    #     iid_part.client_dict,
    #     class_num=num_classes,
    #     verbose=False,
    #     file=csv_file,
    # )
    #
    # iid_part_df = pd.read_csv(csv_file, header=1)
    # iid_part_df = iid_part_df.set_index("client")
    # for col in col_names:
    #     iid_part_df[col] = (iid_part_df[col] * iid_part_df["Amount"]).astype(int)
    #
    # # select first 10 clients for bar plot
    # iid_part_df[col_names].iloc[:10].plot.barh(stacked=True)
    # # plt.tight_layout()
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # plt.xlabel("sample num")
    # plt.savefig(
    #     f"data/CIFAR10/cifar10_iid_clients_10.png", dpi=400, bbox_inches="tight"
    # )
    # dict_users_train_iid = trainset_sampling_label(
    #     args, trainset, trainset_sample_rate, rare_class_nums, iid_part
    # )
    # dict_users_test_iid = testset_sampling(args, testset, number_perclass, iid_part_df)
    #
    # training_number = {j: {} for j in range(args.K)}
    #
    # for i in range(args.K):
    #     training_number[i] = {j: 0 for j in range(num_classes)}
    #     label_class = set(
    #         np.array(trainset.targets)[list(dict_users_train_iid[i])].tolist()
    #     )
    #     # print(list(label_class))
    #     for k in label_class:
    #         training_number[i][k] = list(
    #             np.array(trainset.targets)[list(dict_users_train_iid[i])]
    #         ).count(k)
    #
    # df_training_number = []
    # df_training_number = pd.DataFrame(df_training_number)
    # for i in range(args.K):
    #     temp = pd.Series(training_number[i])
    #     df_training_number[i] = temp
    #
    # df_training_number["Col_sum"] = df_training_number.apply(lambda x: x.sum(), axis=1)
    # df_training_number.loc["Row_sum"] = df_training_number.apply(lambda x: x.sum())

    print("model:", args.model)
    total_params = sum(p.numel() for p in specf_model.parameters())
    print("parameters:", total_params)

    syn_dst = SyntheticCIFAR10()
    syn_noniid_labeldir_part = CIFAR10Partitioner(
        trainset.targets,
        num_clients=num_clients,
        balance=None,
        partition="shards",
        num_shards=200,
        seed=1,
    )
    syn_dict_users = trainset_sampling_label(
        args,
        syn_dst,
        trainset_sample_rate,
        rare_class_nums,
        syn_noniid_labeldir_part,
    )
    serverz = server.Server(
        # args, specf_model, trainset, dict_users_train
        args,
        specf_model,
        trainset,
        dict_users_train,
        syn_dst,
        syn_dict_users,
    )  # dict_users指的是user的local dataset索引
    print("global_model:", serverz.nn.state_dict)

    server_feature = copy.deepcopy(serverz)

    if Train_model:
        (
            global_modelfa,
            similarity_dictfa,
            client_modelsfa,
            loss_dictfa,
            clients_indexfa,
            acc_listfa,
        ) = server_feature.fedfa_anchorloss(
            testset,
            dict_users_test[0],
            similarity=similarity,
            test_global_model_accuracy=True,
        )
    else:
        if similarity:
            similarity_dictfa = torch.load(
                "results/Test/label skew/cifar10/fedfa/seed{}/similarity_dictfa_{}E_{}class.pt".format(
                    args.seed, args.E, C
                )
            )
        acc_listfa = torch.load(
            "results/Test/label skew/cifar10/fedfa/seed{}/acc_listfa_{}E_{}class.pt".format(
                args.seed, args.E, C
            )
        )
        global_modelfa = server_feature.nn
        client_modelsfa = server_feature.nns
        path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/global_model_fedfa_{}E_{}class".format(
            args.seed, args.E, C
        )
        global_modelfa.load_state_dict(torch.load(path_fedfa))
        for i in range(args.K):
            path_fedfa = "results/Test/label skew/cifar10/fedfa/seed{}/client{}_model_fedfa_{}E_{}class".format(
                args.seed, i, args.E, C
            )
            client_modelsfa[i] = copy.deepcopy(global_modelfa)
            client_modelsfa[i].load_state_dict(torch.load(path_fedfa))

    # if save_models:
    #     result_checkpoint_path = results_path + "cifar10_plain_model.ckpt"
    #     torch.save(
    #         {
    #             "model": global_modelfa.state_dict(),
    #             "acc_list": acc_listfa,
    #             "loss_dict": loss_dictfa,
    #         },
    #         result_checkpoint_path,
    #     )
    #     print("Finished. Saving the model and ML metrics...")


if __name__ == "__main__":
    print(f"setting: {int(args.C * args.K)}/{args.K} active clients")

    start_time = time.time()
    run_FedFA()
    end_time = time.time()
    print("Execution Time: ", end_time - start_time)
