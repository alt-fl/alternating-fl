from copy import deepcopy


def he_aggregate(client_idxs, enc_params_dict, client_data_idxs):
    # compute the total number of data samples across all clients
    num_samples = sum(map(lambda k: len(client_data_idxs[k]), client_idxs))
    enc_params = {}
    # for each layer, and the parameters in that layer for all client
    for name, client_params in enc_params_dict.items():
        # for each client's parameters in the layer
        for k, enc_param in client_params.items():
            weight = len(client_data_idxs[k]) / num_samples
            if name not in enc_params:
                # handle special when some layers are not encrypted
                enc_params[name] = enc_param.mul(weight)
                continue
            enc_params[name] = enc_params[name].add(enc_param.mul(weight))
        # serialize ciphertexts
        enc_params[name] = enc_params[name].serialize()
    return enc_params


def fedavg_aggregate(client_idxs, models, client_data_idxs):
    # compute the total number of data samples across all clients
    num_samples = sum(map(lambda k: len(client_data_idxs[k]), client_idxs))
    # we don't care about the client id
    models = list(models.values())
    agg_model = {}

    for id in client_idxs:
        params = models[id]
        for key in params:
            if key not in agg_model:
                agg_model[key] = 0
            agg_model[key] += params[key] * (len(client_data_idxs[id]) / num_samples)
    return agg_model
