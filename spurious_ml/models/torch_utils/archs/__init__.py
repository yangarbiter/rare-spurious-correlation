from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Vgg16Norm02
from .mlps import VggMLP, MLP, MLPv2, LargeMLP, LargeMLPv2
from .alt_resnet import (
    altResNet20,
    altResNet20Norm02,
    altResNet32,
    altResNet32Norm02,
    altResNet110,
    altResNet110Norm02,
)


class CNN002(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_features, n_classes, drop=0.5, n_channels=1, save_intermediates=False):
        super(CNN002, self).__init__()

        self.num_channels = n_channels

        self.save_intermediates = save_intermediates
        self.intermediates = []

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, 200)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(200, 200)),
            ('relu2', activ),
            ('fc3', nn.Linear(200, n_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def get_repr(self, x, mode="cnn_fet"):
        if mode == "cnn_fet":
            features = self.feature_extractor(x)
            return features.view(-1, 64 * 4 * 4)
        elif mode == "last":
            x = self.feature_extractor(x)
            x = x.view(-1, 64 * 4 * 4)
            x = self.classifier.fc1(x)
            x = self.classifier.relu1(x)
            x = self.classifier.drop(x)
            x = self.classifier.fc2(x)
            x = self.classifier.relu2(x)
            return x
        else:
            raise ValueError()

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

class CNN003(nn.Module):
    """https://github.com/yaodongyu/TRADES/blob/e20f7b9b99c79ed3cf0d1bb12a47c229ebcac24a/models/small_cnn.py#L5"""
    def __init__(self, n_features, n_classes, drop=0.5, n_channels=1, save_intermediates=False):
        super(CNN002, self).__init__()

        self.num_channels = n_channels

        self.save_intermediates = save_intermediates
        self.intermediates = []

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 32, 3)),
            ('relu1', activ),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(64 * 4 * 4, n_classes)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def get_repr(self, x, mode="cnn_fet"):
        if mode == "cnn_fet":
            features = self.feature_extractor(x)
            return features.view(-1, 64 * 4 * 4)
        elif mode == "last":
            x = self.feature_extractor(x)
            x = x.view(-1, 64 * 4 * 4)
            x = self.classifier.fc1(x)
            x = self.classifier.relu1(x)
            x = self.classifier.drop(x)
            x = self.classifier.fc2(x)
            x = self.classifier.relu2(x)
            return x
        else:
            raise ValueError()

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features.view(-1, 64 * 4 * 4))
        return logits

