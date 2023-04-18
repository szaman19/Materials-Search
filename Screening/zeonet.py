import torch
import torch.nn as nn
from torch.nn.functional import pad

class BasicModel(nn.Module):
    def __init__(self, features=1, channels=1, dropout=0.5):
        super(BasicModel, self).__init__()
        pad = dict(padding='same')
        p = dropout
        shrink = lambda: nn.MaxPool3d(kernel_size=2, stride=2)
        # 32^3 -> 32^3x32
        self.layer1 = nn.Sequential(
            nn.Conv3d(channels, 32, kernel_size=3, stride=1, **pad, padding_mode='circular'),
            nn.LayerNorm((32, 32, 32, 32)),
            nn.Dropout(p),
            nn.LeakyReLU(0.2),
        )
        # 32^3x32 -> 16^3x64
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, **pad, padding_mode='circular'),
            nn.LayerNorm((64, 32, 32, 32)),
            nn.LeakyReLU(0.2),
            shrink(),
        )
        # 16^3x64 -> 8^3x128
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, **pad, padding_mode='circular'),
            nn.LayerNorm((128, 16, 16, 16)),
            nn.LeakyReLU(0.2),
            shrink(),
        )
        # 8^3x128 -> 4^3x256
        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, **pad, padding_mode='circular'),
            nn.LayerNorm((256, 8, 8, 8)),
            nn.LeakyReLU(0.2),
            shrink(),
        )
        # 4^3x256 -> 512 -> features
        self.fc = nn.Sequential(
            nn.Linear(4**3*256, 512),
            nn.ReLU(),
            nn.Linear(512, features),
            nn.ReLU(),
        )
        for layer in self.fc:
            if type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight)

    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
