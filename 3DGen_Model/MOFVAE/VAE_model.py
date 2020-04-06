import numpy as np 
import torch 
import torch.autograd as autograd
import torch.nn as nn 
from torch.autograd import Variable 
import util

#cuda = True if torch.cuda.is_available() else False 
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, Num_features, Z_dim):
		super(Encoder, self).__init__()
		self.Num_features = Num_features
		self.Z_dim = Z_dim

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

			nn.Conv3d(self.Z_dim, self.Z_dim, 2,2,0), #2
			)
		self.mu_fc = nn.Linear(self.Z_dim, self.Z_dim)
		self.logvar_fc = nn.Linear(self.Z_dim, self.Z_dim)


	def encode(self, x):
		#Input: X 
		#Shape: (Num_batches, Num_features, D, W, H)
		#Output: Mu, Logvar 
		#Shape: (Num_Batches, Z_dim), (Num_Batches, Z_dim)
		x = self.main(x)
		x = x.view(-1, x.size(1))
		return self.mu_fc(x), self.logvar_fc(x)

	
	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	
	def forward(self, x):
		mu, logvar = self.encode(x)
		return self.reparametrize(mu, logvar), mu, logvar 
		

class Decoder(nn.Module):
	"""docstring for Decoder"""
	def __init__(self, z_dimension, num_features, voxel_side_length):
		super(Decoder, self).__init__()
		
		self.z_dimension = z_dimension
		self.voxel_side_length = voxel_side_length
		self.num_features = num_features

		#Input: Encoded_z 
		#Shape(Num_Batches, z_dimension , 1 , 1 , 1)
		#Output: Reconstructed_X 
		#Shape:  (Num_Batches, num_features, voxel_side_length, voxel_side_length,voxel_side_length)

		if(self.voxel_side_length == 32):
			self.main = nn.Sequential(
				
				nn.ConvTranspose3d(self.z_dimension, self.num_features*8, 4,2,0), # self.num_features*8 x 4 x 4 x 4 
				nn.BatchNorm3d(self.num_features*8),
				nn.Tanh(),

				nn.ConvTranspose3d(self.num_features*8, self.num_features*4, 4,2,1), # self.self.num_features*4 x 8 x 8 x 8
				nn.BatchNorm3d(self.num_features*4),
				nn.Tanh(),

				nn.ConvTranspose3d(self.num_features*4, self.num_features*2, 4,2,1), # self.self.num_features*2 x 16 x 16 x 16
				nn.BatchNorm3d(self.num_features*2),
				nn.Tanh(),

				nn.ConvTranspose3d(self.num_features*2, self.num_features, 4,2,1), # self.self.num_features x 32 x 32 x 32 
				nn.Sigmoid(),
			)
		else:
			print("Not Implemented variable sized grid initializer. set voxel_side_length = 32. ")

	def forward(self, x):
		return self.main(x)


if __name__ == '__main__':
	enc = Encoder(Num_features=12, Z_dim = 64)
	dec = Decoder(num_features = 12, z_dimension = 64, voxel_side_length=32)

	data = Variable(torch.rand(2,12,32,32,32))

	z, mu, logvar = enc(data)

	print(z.shape, mu.shape, logvar.shape)

	X = z.view(z.size(0), z.size(1), 1,1,1)
	print(X.shape)

	decoded_z = dec(X)

	print(decoded_z[0].shape)

	chans = range(0,4)
	util.Visualize_4DTensor(decoded_z[0].cpu().detach().numpy(), chans)


		
