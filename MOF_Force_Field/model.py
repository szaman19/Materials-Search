from dataloader import MOFDataset
from tqdm import tqdm
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch_geometric.data import Data, DataLoader
from torch import nn
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_add_pool as gaddp

from MOLGCN import MOLGCN




class MOF_Net(torch.nn.Module):
    def __init__(self,
                 input_features, 
                 mlp = None):
        super(MOF_Net, self).__init__()
        if (mlp):
            self.nn = mlp
        else:
            self.nn = nn.Sequential(
                nn.Linear(input_features, 16),
                nn.BatchNorm1d(16),
                nn.Dropout(.5),
                nn.ReLU(),
                nn.Linear(16,1))
        self.conv = MOLGCN(self.nn)
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        # print(edge_attr.shape)
        x = self.conv(x, edge_index, edge_attr)
#         print(x.shape)
        x = gaddp(x, batch) 
        x = torch.mean(x, dim=1)
#         print(x.shape)
        return x





def run(loader,
		model,
		optimizer,
		loss_func,
		device, 
		train = True):
	
	average_batch_loss = 0 

	def run_():
		total = 0
		desc = ['validation', 'training']
		for data in loader:
			data = data.to(device)
			y_out = model(data)
			y = data.y.to(device)
			loss = loss_func(y, y_out)

			if (train):
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			total += loss.item()
		return total / len(loader)

	
	if (train):
		average_batch_loss = run_()
	else:
		with torch.no_grad(): #This reduces memory usage 
			average_batch_loss = run_()
	return average_batch_loss

		
	
if __name__ == '__main__':
	
	dataset = MOFDataset('FIGXAU_V2.csv','.')
	dataset = dataset.shuffle()
	
	batch_size = 16

	one_tenth_length = int(len(dataset) * 0.1)
	train_dataset = dataset[:one_tenth_length * 8]
	train_loader = DataLoader(train_dataset, batch_size=batch_size)

	val_dataset = dataset[one_tenth_length * 8 :]
	val_loader = DataLoader(val_dataset, batch_size = batch_size)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = MOF_Net(9).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
	loss_func = nn.MSELoss()


	for epoch in range(10):
		print("*" * 100)
		training_loss = run(train_loader,model,optimizer,loss_func,device,True)
		val_loss = run(val_loader,
    						model,
    						optimizer,
    						loss_func,
    						device,
    						False)

		print('\n')
		print("Epoch {} : Training Loss: {:.4f} \t Validation Loss: {:.4f} ".format(epoch + 1, training_loss, val_loss))
		print('\n')

 



