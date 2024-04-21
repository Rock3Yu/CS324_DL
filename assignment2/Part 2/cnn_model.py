# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from torch.nn import Module, Conv2d, MaxPool2d, Linear, CrossEntropyLoss, Sequential, ReLU, BatchNorm2d


class CNN(Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()
        self.features = Sequential(
            Conv2d(3, 64, 3, 1, 1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1),
            Conv2d(64, 128, 3, 1, 1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1),
            Conv2d(128, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            Conv2d(256, 256, 3, 1, 1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1),
            Conv2d(256, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            Conv2d(512, 512, 3, 1, 1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(3, 2, 1),
        )
        self.classifier = Linear(512, 10)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
