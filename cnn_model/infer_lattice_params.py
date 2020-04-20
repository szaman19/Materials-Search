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

 	def forward(self, x):
 		x = self.main(x)
 		x = x.view(-1, x.size(0))
 		return x 


def main():
	epoch = 100
	batch_size = 32 
	learning_rate = 1e-4

	num_features = 12 
	grid_size = 32 

	batch_size = 16

	train_loader = MOFDataset.get_data_loader("../3D_Grid_Data/Test_MOFS.p", batch_size)

	for batch, data in enumerate(train_loader):
		print(batch, type(data[0]), data[0].shape,type(data[1]), type(data[2]))
		print(data[2])
		break


if __name__ == '__main__':

	main()
 		 
