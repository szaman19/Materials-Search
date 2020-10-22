import numpy as np 
import torch 
import torch.autograd as autograd
import torch.nn as nn 
from torch.autograd import Variable 


class VAE(nn.Module):
	"""docstring for VAE"""
	def __init__(self, z_dimension, num_features, voxel_side_length):
		super(VAE, self).__init__()
		self.z_dim = z_dimension
		self.voxel_side_length = voxel_side_length
		self.num_features = num_features

		self.encoder = nn.Sequential(
			# (in_channels, out_channels, kernel_size, stride, padding)
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
		self.mu_fc = nn.Linear(self.z_dim, self.z_dim)
		self.logvar_fc = nn.Linear(self.z_dim, self.z_dim) 

		if(self.voxel_side_length == 32):
			self.decoder = nn.Sequential(
				
				nn.ConvTranspose3d(self.z_dim, self.num_features*8, 4,2,0), # self.num_features*8 x 4 x 4 x 4 
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

	def reparametrize(self, mu, logvar):
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		return mu + eps * std
	def encode(self, x):
		#Input: X 
		#Shape: (Num_batches, Num_features, D, W, H)
		#Output: Mu, Logvar 
		#Shape: (Num_Batches, Z_dim), (Num_Batches, Z_dim)
		x = self.encoder(x)
		x = x.view(-1, x.size(1))
		mu = self.mu_fc(x)
		logvar = self.logvar_fc(x)
		return self.reparametrize(mu,logvar), mu, logvar

	def decode(self, x):
		return self.decoder(x)

	def forward(self, data):
		z, mu, logvar = self.encode(data)
		return self.decode(z), mu, logvar


		
class MLP_VAE(nn.Module):
	"""docstring for MLP_VAE"""
	def __init__(self, z_dim):
		super(MLP_VAE, self).__init__()
		self.z_dim = z_dim

		self.encoder = nn.Sequential(
            nn.Linear(self.z_dim, 400),
            nn.ReLU(True))

		self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(True),
            nn.Linear(400, 28 * 28), 
            nn.Sigmoid())
    
    self.mu = nn.Linear(400,20)
    self.logvar = nn.Linear(400,20)

	def encode(self, x):
	  h1 = self.encoder(x)
	  return self.mu(h1), self.logvar(h1)
	
	def reparameterize(self, mu, logvar):
	  std = torch.exp(0.5*logvar)
	  eps = torch.randn_like(std)
	  return mu + eps*std
	def decode(self, z):
	  h3 = self.decoder(z)
	  return h3
	def forward(self,x):
	  mu, logvar = self.encode(x)
	  z = self.reparameterize(mu, logvar)
	  return self.decode(z), mu, logvar
		
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

	#chans = range(0,4)
	#util.Visualize_4DTensor(decoded_z[0].cpu().detach().numpy(), chans)


		
