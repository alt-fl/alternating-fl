def he_aggregate(client_idxs, enc_params_dict, client_data_idxs):
    # compute the total number of data samples across all clients
    num_samples = sum(map(lambda k: len(client_data_idxs[k]), client_idxs))
    enc_params = {}

    # for each layer, and the parameters in that layer for all client
    for k, client_params in enc_params_dict.items():
        for name, enc_param in client_params.items():
            # for each client's parameters in the layer
            weight = len(client_data_idxs[k]) / num_samples
            if name not in enc_params:
                # for the first client, just set weighted parameters
                enc_params[name] = enc_param.mul(weight)
            else:
                enc_params[name] = enc_params[name].add(enc_param.mul(weight))
        # serialize ciphertexts or not, depending on model assumption...
        # enc_params[name] = enc_params[name].serialize()
    return enc_params


def fedavg_aggregate(client_idxs, models, client_data_idxs):
    # compute the total number of data samples across all clients
    num_samples = sum(map(lambda k: len(client_data_idxs[k]), client_idxs))
    agg_model = {}

    for id in client_idxs:
        params = models[id]
        for key in params:
            if key not in agg_model:
                agg_model[key] = 0
            agg_model[key] += params[key] * (len(client_data_idxs[id]) / num_samples)
    return agg_model
