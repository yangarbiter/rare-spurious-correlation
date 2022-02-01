"""
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.fc(x)
        return x

class MLPv2(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(MLPv2, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.hidden2 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.fc(x)
        return x

class LargeMLP(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLP, self).__init__()
        self.hidden = nn.Linear(n_features[0], 256)
        self.hidden2 = nn.Linear(256, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x

class LargeMLPv2(nn.Module):
    """Basic MLP architecture."""

    def __init__(self, n_features, n_classes, n_channels=None):
        super(LargeMLPv2, self).__init__()
        self.hidden = nn.Linear(n_features[0], 512)
        self.hidden2 = nn.Linear(512, 512)
        self.hidden3 = nn.Linear(512, 512)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.fc(x)
        return x


class VggMLP(nn.Module):

    def __init__(self, n_features, n_classes, n_channels=None):
        super(VggMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features[0], 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, n_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
