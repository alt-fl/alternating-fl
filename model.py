from torch import nn
import torch.nn.functional as F


class ClientModel(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.args = args
        self.name = name

        match self.name.lower():
            case "cifar100":
                self.n_cls = 100
                self.conv1 = nn.Conv2d(
                    in_channels=3, out_channels=6, kernel_size=5, padding=2
                )
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(576, 240)
                self.fc2 = nn.Linear(
                    240, self.args.dims_feature
                )  # args.dims_feature = 168
                self.classifier = nn.Linear(self.args.dims_feature, self.n_cls)
            case "cifar10":
                self.n_cls = 10
                self.conv1 = nn.Conv2d(
                    in_channels=3, out_channels=6, kernel_size=5, padding=2
                )
                self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
                self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(576, 120)
                self.fc2 = nn.Linear(
                    120, self.args.dims_feature
                )  # args.dims_feature = 84
                self.classifier = nn.Linear(self.args.dims_feature, self.n_cls)
            case _:
                raise ValueError(f"{self.name!r} is not a supported dataset")

    def forward(self, x):
        match self.name.lower():
            case "cifar100":
                x = F.sigmoid(self.conv1(x))
                x = self.pool1(x)
                x = F.sigmoid(self.conv2(x))
                x = self.pool2(x)
                x = x.view(x.shape[0], -1)
                x = F.sigmoid(self.fc1(x))
                y_feature = F.sigmoid(self.fc2(x))
                x = self.classifier(y_feature)
            case "cifar10":
                x = F.sigmoid(self.conv1(x))
                x = self.pool1(x)
                x = F.sigmoid(self.conv2(x))
                x = self.pool2(x)
                x = x.view(x.shape[0], -1)
                x = F.sigmoid(self.fc1(x))
                y_feature = F.sigmoid(self.fc2(x))
                x = self.classifier(y_feature)
            case _:
                raise ValueError(f"{self.name!r} is not a supported dataset")

        return y_feature, x
