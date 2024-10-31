#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torch
from torch import nn
import torch.nn.functional as F
import copy
import numpy as np


import copy
import sys

sys.path.append("..")

from utils.global_test import globalmodel_test_on_specifdataset


def fhe_aggregate(client_index, enc_params_dict, dict_users):
    s = sum(map(lambda k: len(dict_users[k]), client_index))
    enc_params = {}
    for name, client_params in enc_params_dict.items():
        for k, enc_param in client_params.items():
            weight = len(dict_users[k]) / s
            if name not in enc_params:
                enc_params[name] = enc_param.mul(weight)
                continue
            enc_params[name].add(enc_param.mul(weight))
        enc_params[name] = enc_params[name].serialize()
    return enc_params


def aggregation(client_index, global_model, client_models, dict_users, fedbn=False):
    s = 0
    for j in client_index:
        # normal
        s += len(dict_users[j])  # local dataset size

    global_w = global_model.state_dict()

    if fedbn:
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()
            if i == 0:
                for key in net_para:
                    if "bn" not in key:
                        global_w[key] = net_para[key] * (len(dict_users[j]) / s)
            else:
                for key in net_para:
                    if "bn" not in key:
                        global_w[key] += net_para[key] * (len(dict_users[j]) / s)
    else:
        for i, j in enumerate(client_index):
            net_para = client_models[j].state_dict()
            if i == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * (len(dict_users[j]) / s)
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * (len(dict_users[j]) / s)

    global_model.load_state_dict(global_w)
