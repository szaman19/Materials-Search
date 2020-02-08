import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import Set2Set
from torch_geometric.nn import GraphConv as GraphConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
import pickle

import matplotlib.pyplot as plt

import numpy as np
import MOFDataset
import Net

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	tdl = MOFDataset.MOFDataset(train=True).get_data()

main()