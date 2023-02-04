
import torch
import torch.nn as nn
from torch.nn.functional import pad

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        # 32^3 -> 16^3x32
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=1, padding=2, padding_mode='circular'),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        # 16^3x32 -> 8^3x32
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(p=0.02))
        # 8^3x32 -> 4^3x32
        self.layer3 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout(p=0.02))
        # 4^3x32 (2048) -> 64
        self.fc1 = nn.Linear(4**3*32, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        # 64 -> 1
        self.fc2 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
