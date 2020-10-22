import numpy as np 
import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torch.utils.data import DataLoader

class MLPAutoEncoder(nn.Module):
	"""docstring for AutoEncoder"""
	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = nn.Sequential(
	    	nn.Linear(32*32*32*12, 512),
	    	nn.ReLU(True),
	    	nn.BatchNorm1d(512),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.BatchNorm1d(512), 
			nn.Linear(512, 256), 
			nn.ReLU(True), 
			nn.BatchNorm1d(256),
			nn.Linear(256, 128))

		self.decoder = nn.Sequential(
	    	nn.Linear(128, 256),
			nn.ReLU(True),
			nn.BatchNorm1d(256),
			nn.Linear(256, 512),
			nn.ReLU(True),
			nn.BatchNorm1d(512),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.BatchNorm1d(512), 
			nn.Linear(512, 32*32*32*12), 
			nn.Tanh())

	def forward(self, x):
		u = self.encoder(x)
		x_prime = self.decoder(u)
		return x_prime

class ConvolutionalAE(nn.Module):
	"""docstring for ConvolutionalAE"""
	def __init__(self, z_dimension, num_features):
		super(ConvolutionalAE, self).__init__()
		self.z_dim = z_dimension
		self.num_features = num_features

		self.encoder = nn.Sequential(
			nn.Conv3d(self.num_features, self.z_dim // 8, 4,2,1), #32
			nn.BatchNorm3d(self.z_dim // 8),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.z_dim // 8, self.z_dim // 4, 4,2,1), #16
			nn.BatchNorm3d(self.z_dim // 4),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.z_dim // 4, self.z_dim // 2, 4,2,1), #4
			nn.BatchNorm3d(self.z_dim // 2),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.z_dim // 2, self.z_dim, 4,2,1), #2
			nn.BatchNorm3d(self.z_dim),
			nn.LeakyReLU(0.2),

			nn.Conv3d(self.z_dim, self.z_dim, 2,2,0), #1
			)

		self.decoder = nn.Sequential(
				
			nn.ConvTranspose3d(self.z_dim, self.num_features*16, 4,2,0), # self.num_features*8 x 4 x 4 x 4 
			nn.BatchNorm3d(self.num_features*16),
			nn.Tanh(),

			nn.ConvTranspose3d(self.num_features*16, self.num_features*4, 4,2,1), # self.self.num_features*4 x 8 x 8 x 8
			nn.BatchNorm3d(self.num_features*4),
			nn.Tanh(),

			nn.ConvTranspose3d(self.num_features*4, self.num_features*2, 4,2,1), # self.self.num_features*2 x 16 x 16 x 16
			nn.BatchNorm3d(self.num_features*2),
			nn.Tanh(),

			nn.ConvTranspose3d(self.num_features*2, self.num_features, 4,2,1), # self.self.num_features x 32 x 32 x 32 
			nn.Sigmoid(),
		)
	def forward(self, x):
		u = self.encoder(x)
		x_prime = self.decoder(u)
		return x_prime




		





# def main():



# if __name__ == '__main__':
# 	main()
# 		