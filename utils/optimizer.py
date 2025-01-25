import copy
import numpy as np
import torch, os, random
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .global_test import test_on_globaldataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def seed_torch(seed, test=False):
    if test:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def fedfa_cl_optimizer(
    args,
    anchorloss_func,
    client_model,
    global_model,
    global_round,
    target_acc,
    dataset_train,
    dict_user,
    testset,
):

    seed_torch(seed=args.seed)

    Dtr = DataLoader(
        DatasetSplit(dataset_train, dict_user), batch_size=args.B, shuffle=True
    )
    label_set = set(np.array(dataset_train.targets)[list(dict_user)].tolist())
    misslabel_set = {i for i in range(args.num_classes)} - label_set

    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, global_round)
    else:
        lr = args.lr
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(client_model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(
            client_model.parameters(), lr=lr, momentum=args.momentum, weight_decay=0.001
        )  #
    # optimizer_c = torch.optim.SGD(client_model.classifier.parameters(), lr=0.1)
    optimizer_c = torch.optim.Adam(client_model.classifier.parameters())

    loss_function = nn.CrossEntropyLoss().to(args.device)
    anchorloss = anchorloss_func.to(args.device)
    epoch_mean_anchor = copy.deepcopy(anchorloss.anchor.data)
    # for i in label_set:
    #     epoch_mean_anchor[i] = torch.zeros_like(epoch_mean_anchor[i])
    # anchorloss_opt = torch.optim.SGD(anchorloss.parameters(),lr=0.001)#, momentum=0.99
    epoch_loss = []

    epoch = 0
    acc = 0
    # for epoch in range(args.E):
    while epoch < args.E and acc < target_acc:
        batch_loss = []
        client_model.train()
        batch_mean_anchor = torch.zeros_like(anchorloss.anchor.data).to(args.device)
        for batch_idx, (imgs, labels) in enumerate(Dtr):

            # predict
            imgs = imgs.to(args.device)
            labels = labels.type(torch.LongTensor).to(args.device)
            features, y_preds = client_model(imgs)

            ys = labels.float()
            loss_anchor = anchorloss(features, ys, Lambda=args.lambda_anchor)
            # print(loss_anchor)

            loss = loss_function(y_preds, labels) + loss_anchor

            # anchorloss_opt.zero_grad()
            optimizer.zero_grad()
            # optimizer_c.zero_grad()

            loss.backward()
            # anchorloss_opt.step()
            optimizer.step()
            # optimizer_c.step()

            C = torch.arange(0, args.num_classes).to(args.device)
            x_c = copy.deepcopy(anchorloss.anchor.data.detach()).to(args.device)
            # miss_label_set = list(set(range(0,args.num_classes))-label_set)
            # C = C[miss_label_set]
            # x_c = x_c[miss_label_set]

            y_c = client_model.classifier(x_c)
            loss_c = loss_function(y_c, C)

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            # #memorize batch feature anchor of last epoch
            for i in set(labels.tolist()):
                batch_mean_anchor[i] += torch.mean(features[labels == i], dim=0)
            #                 batch_mean_anchor[i] = torch.mean(updated_features[labels==i],dim=0)

            #                 #compute epoch mean anchor according to batch mean anchor
            #                 lambda_momentum = 0.99
            #                 epoch_mean_anchor[i] = lambda_momentum*epoch_mean_anchor[i] + (1-lambda_momentum)*batch_mean_anchor[i]

            # anchorloss.anchor.data = epoch_mean_anchor

            if args.verbose and batch_idx % 6 == 0:
                print("loss_anchor: {}".format(loss_anchor))

                print(
                    "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                        epoch,
                        batch_idx * len(imgs),
                        len(Dtr.dataset),
                        100.0 * batch_idx / len(Dtr),
                        loss.item(),
                    )
                )
            batch_loss.append(loss.item())

        # if epoch == 0:
        #     for i in label_set:
        #         #compute batch mean anchor according to batch label
        #         batch_mean_anchor[i] = batch_mean_anchor[i]/(batch_idx+1)
        #         epoch_mean_anchor[i] = batch_mean_anchor[i]
        # else:
        for i in label_set:
            # compute batch mean anchor according to batch label
            batch_mean_anchor[i] = batch_mean_anchor[i] / (batch_idx + 1)

            # compute epoch mean anchor according to batch mean anchor
            lambda_momentum = args.momentum_anchor  # pow(2, -(epoch+1))
            epoch_mean_anchor[i] = (
                lambda_momentum * epoch_mean_anchor[i]
                + (1 - lambda_momentum) * batch_mean_anchor[i]
            )

        # anchorloss.anchor.data = epoch_mean_anchor

        # memorize epoch loss
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # test accuracy, if testset is given
        if testset:
            client_model.eval()
            acc, _ = test_on_globaldataset(args, client_model, testset)
            acc = acc / 100
        epoch += 1

    anchorloss.anchor.data = epoch_mean_anchor

    return anchorloss, client_model, epoch_loss, epoch + 1
