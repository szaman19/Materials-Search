import torch
from torch_geometric.data import Data, DataLoader
from model import MOF_Net, run
from MOLGCN import MOLGCN
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.data as data_utils

def gaussian_dist(val, mean, variance):
    temp = - ((val-mean)**2) / (2 * (variance))
    return (1 / (np.sqrt(2 * np.pi* variance)))* np.exp(temp)

def energy(bond_type, distance):
    if bond_type == 1:
        return gaussian_dist(distance, 0.6, .1)
    elif bond_type == 2:
        return gaussian_dist(distance, 0.05, 0.01) 
    else:
        return gaussian_dist(distance, 0.3, 0.02)

def generate_graph_data(N):
    data_list = []
    for data_sample in range(N):
        node_features = torch.eye(3)
        edge_list = torch.zeros((2,3)).long()
        
        edge_list[0][0] = 0
        edge_list[0][1] = 1
        edge_list[0][2] = 2
        
        edge_list[1][0] = 1
        edge_list[1][1] = 2
        edge_list[1][2] = 0
        
        distances = torch.rand((3,1))
        np_dists = distances.data.cpu().numpy()
        
        edge_features = distances
        y = energy(1,np_dists[0][0]) + energy(2,np_dists[1][0])+ energy(3,np_dists[2][0])
        
        node_features = node_features.float()
        edge_list = edge_list.long()
        y = torch.tensor(y).float()
        edge_features = edge_features.float()
        geom_data = Data(x=node_features, edge_index = edge_list, edge_attr = edge_features ,y = y)
        data_list.append(geom_data)
    return data_list

data_list = generate_graph_data(10000)
loader = DataLoader(data_list, batch_size=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = nn.Sequential(nn.Linear(7,16),
                    nn.ReLU(),
                    nn.Linear(16,1)
                   )
model = MOF_Net(7, mlp).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.MSELoss()

for epoch in range(401):
    total = 0
    for batch, data in enumerate(loader):
#         print(data)
        x = data.to(device)
        out = model(x)
        y = data.y.to(device)
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        total += loss.item()
        optimizer.step()
    if (epoch % 10 == 0):
        print(total / len(loader))