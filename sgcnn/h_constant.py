from torch_geometric.data import DataLoader
from torch import nn
from torch_geometric.nn import CGConv, TopKPooling, GCNConv, NNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gaddp
from torch_geometric.nn import BatchNorm
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, num_features):
        super(Net, self).__init__()

        self.g = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(128,100),
            nn.BatchNorm1d(100),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(100,64),
            nn.BatchNorm1d(64),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.BatchNorm1d(16),
            nn.Dropout(.5),
            nn.ReLU(),
            nn.Linear(16,num_features)
        )
        self.conv1 = NNConv(num_features, num_features, self.g, root_weight=False, bias=False)
        self.gbn1 = BatchNorm(32)

        # channel in is 32, channel out is 32. Ratio is 0.5

        self.pool1 = TopKPooling(32, ratio=0.5)

        self.conv2 = NNConv(num_features, num_features, self.g, root_weight=False, bias=False)
        self.gbn2 = BatchNorm(32)

        # channel in is 32, channel out is 32. Ratio is 0.8
        self.pool2 = TopKPooling(32, ratio=0.8)

        self.conv3 = NNConv(num_features, num_features, self.g, root_weight=False, bias=False)
        self.gbn3 = BatchNorm(32)

        self.pool3 = TopKPooling(32, ratio=0.8)

        self.conv4 = NNConv(num_features, num_features, self.g, root_weight=False, bias=False)

        self.gbn4 = BatchNorm(32)

        self.pool4 = TopKPooling(32, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.bn1 = torch.nn.BatchNorm1d(num_features=128)
        self.lin2 = torch.nn.Linear(128, 16)
        self.bn2 = torch.nn.BatchNorm1d(num_features=16)
        self.lin3 = torch.nn.Linear(16, 1)
        self.activ = nn.Tanhshrink()

    def forward(self, data):
        x, edge_index, batch, weight = data.x, data.edge_index, data.batch, data.weight

        x = self.gbn1(F.relu(self.conv1(x, edge_index, weight)))
        x, edge_index, weight, batch, _, _ = self.pool1(x, edge_index, weight, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gbn2(F.relu(self.conv2(x, edge_index, weight)))
        x, edge_index, weight, batch, _, _ = self.pool2(x, edge_index, weight, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gbn3(F.relu(self.conv3(x, edge_index, weight)))
        x, edge_index, weight, batch, _, _ = self.pool3(x, edge_index, weight, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gbn4(F.relu(self.conv3(x, edge_index, weight)))
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, weight, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        # x = self.set2set(x, batch)
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=.2, training=self.training)

        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        # x = F.dropout(x, p=.5, training = self.training)
        x = self.lin3(x)
        x = x.view(-1)

        return x