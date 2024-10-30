import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tenseal as ts

from pprint import pp


import utils.optimizer as op


def client_fedfa_cl(
    args,
    client_index,
    anchorloss_funcs,
    client_models,
    global_model,
    global_round,
    dataset_train,
    dict_users,
    loss_dict,
    he_context,
    mask,
    enc_params,
):  # update nn
    enc_params_dict = {}

    for k in client_index:  # k is the index of the client
        print("Client {} client_fedfa_anchorloss...".format(k))

        model = client_models[k]

        if enc_params:
            start = time.time()
            for name, param in model.named_parameters():
                if name not in mask:
                    continue
                dec_param = enc_params[name].decrypt()
                param_flat = param.data.view(-1)
                param_flat[mask[name]] = torch.tensor(dec_param)
            end = time.time()
            print(f"\ttotal decryption time: {end - start}s")

        start = time.time()
        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(
            args,
            anchorloss_funcs[k],
            client_models[k],
            global_model,
            global_round,
            dataset_train,
            dict_users[k],
        )
        loss_dict[k].extend(loss)
        end = time.time()
        print(f"\tlocal training time: {end - start}s")

        start = time.time()
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            param_flat = param.data.view(-1)
            enc_param_flat = ts.ckks_vector(
                he_context, param_flat[mask[name]].detach().clone()
            )
            if name not in enc_params_dict:
                enc_params_dict[name] = {}
            enc_params_dict[name][k] = enc_param_flat
            # hide encrypted parameters by settign random values
            param_flat[mask[name]] = -9999  # torch.randn(mask[name].shape)

        end = time.time()
        print(f"\ttotal encryption time: {end - start}s\n")

    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]] * args.E
        loss_dict[j].extend(loss)

    return anchorloss_funcs, client_models, loss_dict, enc_params_dict
