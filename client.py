from copy import deepcopy
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tenseal as ts

from pprint import pp
import tracemalloc
import psutil

import utils.optimizer as op


def client_fedfa_cl(
    args,
    client_index,
    anchorloss_funcs,
    client_models,
    global_model,
    global_round,
    epoch_transition,
    dataset_train,
    dict_users,
    loss_dict,
    he_context,
    secret_key,
    mask,
    enc_params,
    is_auth,
):  # update nn
    # reset peak memory to current memory to start tracking memory usage
    tracemalloc.reset_peak()

    enc_params_dict = {}
    local_training_times = {}

    # we assume the context is serialized
    he_context = ts.context_from(deepcopy(he_context)) if he_context else None

    for k in client_index:  # k is the index of the client
        print("Client {} client_fedfa_anchorloss...".format(k))

        client_models[k] = deepcopy(client_models[k])
        anchorloss_funcs[k] = deepcopy(anchorloss_funcs[k])

        local_training_times[k] = {}
        model = client_models[k]
        # kickstart CPU utilization tracing
        psutil.cpu_percent()

        if enc_params:
            with torch.no_grad():
                start = time.time()
                for name, param in model.named_parameters():
                    if name not in mask:
                        continue
                    dec_param = enc_params[name].decrypt(secret_key)
                    param_flat = param.view(-1)
                    param_flat[mask[name]] = torch.tensor(dec_param).to(args.device)
                end = time.time()
                dec_time = end - start
                local_training_times[k]["decryption"] = dec_time
                print(f"\ttotal decryption time: {dec_time:.2f}s")
        else:
            local_training_times[k]["decryption"] = 0

        # acc = global_acc / 100
        # target_acc = (
        #     acc + args.dyn_epoch_base - args.dyn_epoch_base * acc**args.dyn_epoch_factor
        # )

        epoch = epoch_transition.estimate_epoch(global_round + 1)

        start = time.time()
        anchorloss_funcs[k], client_models[k], loss = op.fedfa_cl_optimizer(
            args,
            anchorloss_funcs[k],
            client_models[k],
            global_model,
            global_round,
            dataset_train,
            dict_users[k],
            epoch,
        )
        loss_dict[k].extend(loss)
        end = time.time()
        training_time = end - start
        local_training_times[k]["training"] = training_time
        print(f"\tlocal training time: {training_time:.2f}s")
        local_training_times[k]["epoch"] = epoch

        if args.epsilon > 0 and is_auth:
            with torch.no_grad():
                start = time.time()
                for name, param in model.named_parameters():
                    if name not in mask:
                        continue
                    param_flat = param.view(-1)
                    enc_param_flat = ts.ckks_vector(
                        he_context, param_flat[mask[name]].cpu().detach().clone()
                    )
                    if name not in enc_params_dict:
                        enc_params_dict[name] = {}
                    enc_params_dict[name][k] = enc_param_flat
                    # hide encrypted parameters by setting random values
                    param_flat[mask[name]] = torch.randn(mask[name].shape).to(
                        args.device
                    )
                end = time.time()
                enc_time = end - start
                local_training_times[k]["encryption"] = enc_time
                print(f"\ttotal encryption time: {enc_time:.2f}s\n")
        else:
            local_training_times[k]["encryption"] = 0

        client_cpu_util = psutil.cpu_percent()
        local_training_times[k]["cpu_util"] = client_cpu_util

    index_nonselect = list(set(i for i in range(args.K)) - set(client_index))
    for j in index_nonselect:
        loss = [loss_dict[j][-1]] * args.E
        loss_dict[j].extend(loss)

    return (
        anchorloss_funcs,
        client_models,
        loss_dict,
        local_training_times,
        enc_params_dict,
    )
