from dataloader import MOFDataset
import os.path as osp 
import os


import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.data as data_utils

from torch_geometric.data import Data, DataLoader
from model import MOF_Net, run
from MOLGCN import MOLGCN
import model



cur_dir = os.curdir
dataset = MOFDataset('FIGXAU_V2.csv','.')


dataset = dataset.shuffle()
batch_size = 2
one_tenth_length = int(len(dataset) * 0.1)

train_dataset = dataset[:one_tenth_length * 8]
train_loader = DataLoader(train_dataset, batch_size=batch_size)

val_dataset = dataset[one_tenth_length * 8 :]
val_loader = DataLoader(val_dataset, batch_size = batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mlp = nn.Sequential(nn.Linear(7,16),
                    nn.ReLU(),
                    nn.Linear(16,1)
                   )

                   
model = MOF_Net(9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.MSELoss()

train_loss_list = []
val_loss_list = [] 


for data in train_loader:
	print(data.x)
	y_pred = model(data)
	break

# for epoch in range(100):
#     training_loss = run(train_loader,model,optimizer,loss_func,device,True)
#     val_loss = run(val_loader,
#                    model,
#                    optimizer,
#                    loss_func,
#                    device,
#                    False)
#     train_loss_list.append(training_loss)
#     val_loss_list.append(val_loss)
#     print("Epoch {} : Training Loss: {:.4f} \t Validation Loss: {:.4f} ".\
#     	format(epoch + 1, training_loss, val_loss)) 
#     break    