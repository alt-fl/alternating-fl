from copy import deepcopy

import torch
from torch.nn import CrossEntropyLoss

from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from opacus import PrivacyEngine

from datasets.syn import IndexedDataset


def optimize(
    args,
    anchorloss,
    model,
    train_data,
    data_idx,
    num_epoch=5,
    comm_round=0,
):
    model.train()

    train_dataloader = DataLoader(
        IndexedDataset(train_data, data_idx),
        batch_size=args.B,
        shuffle=True,
    )

    lr = args.lr
    if args.weight_decay != 0:
        lr = lr * pow(args.weight_decay, comm_round)

    if args.optimizer == "adam":
        optim = Adam(model.parameters(), lr=lr)
    else:
        optim = SGD(
            model.parameters(), lr=lr, momentum=args.momentum, weight_decay=0.001
        )

    privacy_engine = PrivacyEngine()

    # model, optim, train_dataloader = privacy_engine.make_private_with_epsilon(
    #     module=model,
    #     optimizer=optim,
    #     data_loader=train_dataloader,
    #     target_epsilon=5,
    #     target_delta=1 / len(data_idx),
    #     epochs=num_epoch,
    #     max_grad_norm=1.0,
    # )

    # the classifier calibration
    optim_cl = Adam(model.classifier.parameters())

    loss_func = CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss.to(args.device)

    epoch_mean_anchor = deepcopy(anchorloss.anchor.data)

    full_labels = torch.arange(0, args.num_classes).to(args.device)
    epoch_loss = []

    for _ in range(num_epoch):
        batch_loss = []
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data)

        for batch_idx, (imgs, labels) in enumerate(train_dataloader):
            # predict
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = model(imgs.to(args.device))

            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda=args.lambda_anchor)

            loss = loss_func(y_preds, labels) + loss_anchor

            optim.zero_grad()
            loss.backward()
            optim.step()
            optim.zero_grad()

            # compute classifier calibration loss
            x_cl = deepcopy(anchorloss.anchor.data.detach()).to(args.device)
            y_cl = model.classifier(x_cl)
            loss_cl = loss_func(y_cl, full_labels)

            optim_cl.zero_grad()
            loss_cl.backward()
            optim_cl.step()

            # memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels == i], dim=0)
            batch_loss.append(loss.item())

        unique_labels = set(train_data.targets[list(data_idx)])
        for i in unique_labels:
            # compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i] / len(train_dataloader)

            # compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor  # pow(2, -(epoch+1))
            epoch_mean_anchor[i] = (
                lambda_momentum * epoch_mean_anchor[i]
                + (1 - lambda_momentum) * batch_mean_anchor[i]
            )

        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    anchorloss.anchor.data = epoch_mean_anchor

    return epoch_loss
