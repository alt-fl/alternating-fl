import torch
from torch.utils.data import DataLoader
from pprint import pp


import utils.optimizer as op


def get_enc_mask(args, model, dst, dst_idx, ratio=0.1):
    sens_maps = get_sensitivity_maps(args, model, dst, dst_idx)
    agg_map = [
        torch.zeros(p.shape).to(args.device) for _, p in model.named_parameters()
    ]

    for k in range(args.K):
        agg_weight = float(len(dst_idx[k])) / len(dst)
        for layer, sens in enumerate(sens_maps[k]):
            agg_map[layer] += sens * agg_weight

    return get_most_sensitivive(agg_map, model, ratio)


def get_most_sensitivive(sens_map, model, ratio=0.1):
    map_flat = torch.cat(tuple(map(lambda l: l.view(-1), sens_map)))
    top_indices = torch.topk(map_flat, int(len(map_flat) * ratio), largest=True).indices

    last_idx = 0
    enc_map = {}
    for name, param in model.named_parameters():
        total_num = param.numel()
        indices = top_indices[
            (top_indices >= last_idx) & (top_indices < last_idx + total_num)
        ]
        print(f"{name}: {len(indices)}/{total_num}")

        if len(indices) == 0:
            last_idx += total_num
            continue

        enc_map[name] = indices - last_idx
        last_idx += total_num

    return enc_map


def get_sensitivity_maps(args, model, dst, dst_idx):
    sens_maps = []
    for k in range(args.K):
        loader = DataLoader(
            op.DatasetSplit(dst, dst_idx[k]), batch_size=args.B, shuffle=True
        )

        gradients = [None for _ in model.named_parameters()]
        loss_fun = torch.nn.CrossEntropyLoss().to(args.device)
        for img, label in loader:
            img, label = img.to(args.device), label.to(args.device)
            _, y_preds = model(img)
            loss = loss_fun(y_preds, label)
            loss.backward()

            for i, (_, param) in enumerate(model.named_parameters()):
                abs_grads = torch.abs(param.grad.data)
                if gradients[i] is None:
                    gradients[i] = abs_grads
                else:
                    gradients[i] += abs_grads

        weight = 1 / len(loader)

        for sens in gradients:
            sens *= weight
        sens_maps.append(gradients)

    return sens_maps
