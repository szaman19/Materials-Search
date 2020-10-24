from torch_geometric.data import DataLoader
from torch import nn
from dgl.nn.pytorch.glob import Set2Set
from torch_geometric.nn import CGConv, TopKPooling, GCNConv, NNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gaddp
import torch


class Net(torch.nn.Module):
    """docstring for Net"""

    def __init__(self, node_dim=11, edge_dim=5, num_target=8):
        super(Net, self).__init__()

        node_hidden_dim=128
        edge_hidden_dim=128

        self.preprocess = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, node_hidden_dim),
        )
        edge_net = nn.Sequential(
            nn.Linear(node_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, edge_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(128, node_hidden_dim*node_hidden_dim),
        )
        self.conv1 = NNConv(node_hidden_dim, node_hidden_dim, edge_net, aggr='mean', root_weight=False, bias=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 6, 128)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.weight
        x = x.view(x.size(0), -1)
        x = self.conv1(x, edge_index, edge_attr)
        x = gaddp(x, batch)
        return x