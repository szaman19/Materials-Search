import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
#from torch_geometric.nn import GraphConv, TopKPooling
#from torch_geometric.nn import global_mean_pool as gap
#from torch_geometric.nn import global_max_pool as gmp

#import matplotlib.pyplot as plt

import numpy as np
import MOFDataset
import pickle

def main():
	file = open("sparse_test_data_half_precision.p", 'wb')
	training_data_list = MOFDataset.MOFDataset(train=False).get_data()
	pickle.dump(training_data_list, file)

	train_file = open("sparse_train_data_half_precision_sp.p", 'wb')
	dl = MOFDataset.MOFDataset(train=True).get_data()
	pickle.dump(dl, train_file)


main()
