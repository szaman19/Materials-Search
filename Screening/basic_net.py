import math

import torch
import torch.nn as nn
from torch.nn.functional import pad

class BasicModel(nn.Module):
    def __init__(self, features=1, channels=1, depth=32, dropout=0.2):
        super(BasicModel, self).__init__()
        pad = dict(padding='same')
        p = dropout
        n = depth
        # 32^3 -> 16^3x32
        self.layer1 = nn.Sequential(
            nn.Conv3d(channels, n, kernel_size=5, stride=1, **pad, padding_mode='circular'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        # 16^3x32 -> 8^3x32
        self.layer2 = nn.Sequential(
            nn.Conv3d(n, n, kernel_size=3, stride=1, **pad),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(p)
        )
        # 8^3x32 -> 4^3x32
        self.layer3 = nn.Sequential(
            nn.Conv3d(n, n, kernel_size=3, stride=1, **pad),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(p)
        )
        # 4^3x32 (2048) -> 64 -> features
        self.fc = nn.Sequential(
            nn.Linear(4**3*n, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p),
            nn.Linear(64, features),
        )
        for layer in self.fc:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)

    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
