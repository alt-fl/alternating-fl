import torch
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader

from opacus import PrivacyEngine

from datasets.syn import IndexedDataset


def optimize(
    args, model, train_data, data_idx, num_epoch=5, comm_round=0, use_dp=False
):
    """
    Simple FedAvg implementation
    """
    model.train()

    train_dataloader = DataLoader(
        IndexedDataset(train_data, data_idx), batch_size=args.B, shuffle=True
    )

    lr = args.lr
    if args.weight_decay != 0:
        lr = args.lr * pow(args.weight_decay, comm_round)

    if args.optimizer.lower() == "adam":
        optim = Adam(model.parameters(), lr=lr)
    else:
        optim = SGD(
            model.parameters(),
            lr=lr,
            weight_decay=0.0001,
            momentum=args.momentum,
        )

    if use_dp:
        # use DP protected models, note that we do not have persistent privacy
        # accounting with this, but it should be acceptabel due to our assumptions,
        # e.g., cross-silo FL setting and sample-level privacy
        privacy_engine = PrivacyEngine()
        model, optim, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optim,
            data_loader=train_dataloader,
            target_epsilon=args.dp_epsilon,
            # for delta we just set a reasonable default
            target_delta=max(1 / len(data_idx), 1e-5),
            epochs=num_epoch,
            max_grad_norm=args.max_grad_norm,
            batch_first=True,
        )

    loss_func = CrossEntropyLoss().to(args.device)

    epoch_loss = []
    for _ in range(num_epoch):
        batch_loss = []

        for imgs, labels in train_dataloader:
            # predict
            labels = labels.type(torch.LongTensor).to(args.device)
            _, y_preds = model(imgs.to(args.device))

            optim.zero_grad()
            loss = loss_func(y_preds, labels)
            loss.backward()

            optim.step()
            batch_loss.append(loss.item())

        # record epoch loss
        epoch_loss.append(sum(batch_loss) / len(batch_loss))

    return epoch_loss
