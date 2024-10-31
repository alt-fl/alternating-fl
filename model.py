from torch import nn
import torch.nn.functional as F


class ClientModel(nn.Module):
    def __init__(self, args, name):
        super().__init__()
        self.args = args
        self.achitecture = args.model
        self.name = name

        match self.achitecture.lower():
            case "cnn":
                self.n_cls = 10
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
                self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(64 * 5 * 5, 384)
                self.fc2 = nn.Linear(384, args.dims_feature)  # args.dims_feature=192
                self.classifier = nn.Linear(args.dims_feature, self.n_cls)
            case "lenet5":
                self.n_cls = 10
                self.conv1 = nn.Conv2d(
                    in_channels=3, out_channels=6, kernel_size=5, padding=2
                )
                self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
                self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
                self.fc1 = nn.Linear(576, 120)
                self.fc2 = nn.Linear(
                    120, self.args.dims_feature
                )  # args.dims_feature = 84
                self.classifier = nn.Linear(self.args.dims_feature, self.n_cls)
            case _:
                raise ValueError(f"{self.name!r} is not a supported architecture")

    def forward(self, x):
        match self.achitecture.lower():
            case "cnn":
                x = self.pool(F.sigmoid(self.conv1(x)))
                x = self.pool(F.sigmoid(self.conv2(x)))
                x = x.view(-1, 64 * 5 * 5)
                x = F.sigmoid(self.fc1(x))
                y_feature = F.sigmoid(self.fc2(x))
                x = self.classifier(y_feature)
            case "lenet5":
                x = F.sigmoid(self.conv1(x))
                x = self.pool1(x)
                x = F.sigmoid(self.conv2(x))
                x = self.pool2(x)
                x = x.view(x.shape[0], -1)
                x = F.sigmoid(self.fc1(x))
                y_feature = F.sigmoid(self.fc2(x))
                x = self.classifier(y_feature)
            case _:
                raise ValueError(f"{self.name!r} is not a supported architecture")

        return y_feature, x
