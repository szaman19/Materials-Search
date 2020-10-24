<<<<<<< HEAD
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

from pathlib import Path





cur_dir = os.curdir
dataset = MOFDataset('FIGXAU_V2.csv','.')



batch_size = 48

one_tenth_length = int(len(dataset) * 0.1)

train_dataset = dataset[:one_tenth_length * 8]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = dataset[one_tenth_length * 8 :]
val_loader = DataLoader(val_dataset, batch_size = 1024)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp = nn.Sequential(nn.Linear(5,1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024,256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256,32),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(32,1)
                    )


model = MOF_Net(5,mlp).to(device)

PATH = 'warmup_model.pt'

saved_model = Path(PATH)

if (saved_model.is_file()):
    print("Loading Saved Model")
    model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want

model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_func = nn.SmoothL1Loss()

train_loss_list = []
val_loss_list = [] 
# warmup_batch = next(iter(train_loader))

# not_converge = True

# counter = 0
# while(not_converge):
#     total = 0
#     for epoch in range(1000):
#         data = warmup_batch.to(device)
#         y_out = model(data)
#         y = data.y.to(device)
#         loss = loss_func(y,y_out)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
# #         print("Epoch {} : Training Loss: {:.9f} \t".\
# #               format(epoch + 1, loss.item()))
#         total += loss.item()
#     total = total / 1000
#     counter +=1
#     if(total > 2):
#         print("Not Converged. Continuing Training. Average Error: {}".format(total))
#         not_converge = True
#     else:
#         print("Converged! Ending Training. Average Error: {}".format(total))
#         not_converge = False
#     if (counter > 20):
#         not_converge = False
        


for epoch in range(20):
    training_loss = run(train_loader,model,optimizer,loss_func,device,True)
    val_loss = run(val_loader,
                   model,
                   optimizer,
                   loss_func,
                   device,
                   False)
    train_loss_list.append(training_loss)
    val_loss_list.append(val_loss)
    print("Epoch {} : Training Loss: {:.4f} \t Validation Loss: {:.4f} ".\
          format(epoch + 1, training_loss, val_loss)) 

with torch.no_grad():    
    data = warmup_batch.to(device)
    y = model(data)
    print("Predicted: \n \t", y)
    print("Actual: \n \t", data.y)
torch.save(model.state_dict(), PATH)
=======
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
>>>>>>> parent of 8725cd7... Updated dataloader
