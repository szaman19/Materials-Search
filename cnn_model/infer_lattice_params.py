import torch
import torch.nn as nn 
from torch.autograd import Variable 

import sys
# sys.path.append("..")

from mof_dataset import MOFDataset  


class Conv_Model(nn.Module):
 	"""docstring for Conv_Model"""
 	def __init__(self, num_features, z_dim):
 		super(Conv_Model, self).__init__()
 		
 		self.Num_features = num_features
 		self.Z_dim = z_dim		
 		self.main = nn.Sequential(
			nn.Conv3d(self.Num_features, self.Z_dim // 8, 4,2,1), #32
			nn.BatchNorm3d(self.Z_dim // 8),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.Z_dim // 8, self.Z_dim // 4, 4,2,1), #16
			nn.BatchNorm3d(self.Z_dim // 4),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.Z_dim // 4, self.Z_dim // 2, 4,2,1), #4
			nn.BatchNorm3d(self.Z_dim // 2),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.Z_dim // 2, self.Z_dim, 4,2,1), #2
			nn.BatchNorm3d(self.Z_dim),
			nn.LeakyReLU(0.2),

			nn.Conv3d(self.Z_dim, self.Z_dim, 2,2,0), #1
			)

 		self.infer = nn.Sequential(
 			nn.Linear(self.Z_dim, self.Z_dim // 2),
 			nn.ReLU(),
 			nn.Dropout(p=0.2),
 			nn.Linear(self.Z_dim // 2, 3)
 			)

 	def forward(self, x):
 		x = self.main(x)
 		x = x.squeeze()
 		x = self.infer(x)
 		return x 

epoch = 250
batch_size = 32 
learning_rate = 1e-3

num_features = 11 
grid_size = 32 
batch_size = 32

cuda = True if torch.cuda.is_available() else False 


device = torch.device('cuda' if cuda else 'cpu')

model = Conv_Model(11,2048).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

criterion = nn.MSELoss(reduction = 'mean')

train_loader = MOFDataset.get_data_loader("../3D_Grid_Data/Training_MOFS.p", batch_size)
test_loader = MOFDataset.get_data_loader("../3D_Grid_Data/Test_MOFS.p", 512)


# a = None 
# for batch, mof in enumerate(test_loader):
# 	if a is not None:
# 		a = torch.cat((a,mof['lattice_params'].squeeze().detach().clone()), 0)
# 	else:
# 		a = mof['lattice_params'].squeeze().detach().clone()
# 	print(mof['lattice_params'].squeeze().shape) 
# 	print(torch.sum(mof['lattice_params'], axis=0))

# for batch, mof in enumerate(train_loader):
# 	a = torch.cat((a,mof['lattice_params'].squeeze().detach().clone()), 0)
# 	print(mof['lattice_params'].squeeze().shape) 
# 	print(torch.sum(mof['lattice_params'], axis=0))

# print(a.shape)
# print(torch.mean(a, axis=0))
# print(torch.std(a, axis=0))

# print(a / len(test_loader.dataset))
def train():
	total_loss = 0
	for batch, mof in enumerate(train_loader):
		data = mof['data']
		# print(data.shape)
		lattice_params = mof['lattice_params'].to(device) 
		data = data.float().to(device)
		output = model(data)
		loss = criterion(output, lattice_params)
		total_loss +=loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print(batch, loss)
	total_loss = total_loss/ (len(train_loader))
	return total_loss

def test():
	total_loss = 0
	for batch, mof in enumerate(test_loader):
		with torch.no_grad():
			data = mof['data']
			# print(data.shape)
			lattice_params = mof['lattice_params'].to(device) 
			data = data.float().to(device)
			output = model(data)
			loss = criterion(output, lattice_params)
			total_loss +=loss.item()
		# print(batch, loss)
	total_loss = total_loss/ (len(test_loader))
	return total_loss

def main():

	for i in range(epoch):
		training_loss = train()
		print("Epoch {} : Training Loss: {:.4f}".format(i, training_loss))
		if (i % 5== 0):
			test_loss = test()
			print("Epoch {} : Training Loss: {:.4f} Test Loss: {:.4f}".format(i, training_loss, test_loss))

			# break


if __name__ == '__main__':
	# print('Done')
	main()
 		 
