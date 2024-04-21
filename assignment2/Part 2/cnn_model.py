from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, CrossEntropyLoss, Sequential, ReLU


class CNN(nn.Module):
    # def __init__(self, n_channels, n_classes):
    #     super(CNN, self).__init__()
    #     self.layers = Sequential(
    #         Conv2d(in_channels=3, out_channels=64, 3, 1, 1),
    #         MaxPool2d(3, 2, 1),
    #         Conv2d(64, 128, 3, 1, 1),
    #         MaxPool2d(3, 2, 1),
    #         Conv2d(128, 128, 3, 1, 1),
    #         Conv2d(128, 256, 3, 1, 1),
    #         MaxPool2d(3, 2, 1),
    #         Conv2d(256, 512, 3, 1, 1),
    #         Conv2d(512, 512, 3, 1, 1),
    #         MaxPool2d(3, 2, 1),
    #         Conv2d(512, 512, 3, 1, 1),
    #         Conv2d(512, 512, 3, 1, 1),
    #         MaxPool2d(3, 2, 1),
    #         Linear(in_features=512, out_features=10)
    #     )
    #     self.loss_fn = CrossEntropyLoss()
    # 
    # def forward(self, x):
    #     return self.layers(x)

    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.classifier = nn.Linear(512, 10)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
