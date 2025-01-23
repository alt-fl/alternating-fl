import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, name, num_classes=10, factor=1, dims_feature=84):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.dims_feature = dims_feature

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # here we might adjust the sizes depending on the experiment
        self.fc1 = nn.Linear(576, 120 * factor)
        self.fc2 = nn.Linear(120 * factor, self.dims_feature)
        self.classifier = nn.Linear(self.dims_feature, self.num_classes)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = F.sigmoid(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)
        x = F.sigmoid(self.fc1(x))
        y_feature = F.sigmoid(self.fc2(x))
        x = self.classifier(y_feature)

        return y_feature, x
