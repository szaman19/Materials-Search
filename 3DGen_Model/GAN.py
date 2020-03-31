import torch
import torch.nn as nn 
from torch.autograd import Variable 


class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, z_len,cube_side, num_atoms):
		super(Generator, self).__init__()
		self.inp_vector = z_len
		self.length = cube_side
		self.num_channels = num_atoms

		#inp.vector x 1 x 1 x 1
		#Conv3d(in_channels, out_channels, kerbel_size, stride, padding)
		self.main = nn.Sequential(
			
			nn.ConvTranspose3d(self.inp_vector, self.length*8, 4,2,0), # self.length*8 x 4 x 4 x 4 
			nn.BatchNorm3d(512),
			nn.Tanh(),

			nn.ConvTranspose3d(self.length*8, self.length*4, 4,2,1), # self.length*4 x 8 x 8 x 8
			nn.BatchNorm3d(256),
			nn.Tanh(),

			nn.ConvTranspose3d(self.length*4, self.length*2, 4,2,1), # self.length*2 x 16 x 16 x 16
			nn.BatchNorm3d(128),
			nn.Tanh(),

			nn.ConvTranspose3d(self.length*2, self.length, 4,2,1), # self.length x 32 x 32 x 32 
			nn.BatchNorm3d(64),
			nn.Tanh(),

			nn.ConvTranspose3d(self.length, self.num_channels, 4,2,1), #self.num_channels x 64 x 64 x 64
			nn.Sigmoid()
			)
	def forward(self, x):
		x = x.view(x.size(0),x.size(1),1,1,1)
		return self.main(x)

class Discriminator(nn.Module):
	def __init__(self, num_atoms, cube_side):
		super(Discriminator,self).__init__()
		self.num_channels  = num_atoms
		self.length = cube_side

		self.main = nn.Sequential(

			nn.Conv3d(self.num_channels, self.length, 4,2,1), #64
			nn.BatchNorm3d(64),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.length, self.length * 2, 4,2,1), #32
			nn.BatchNorm3d(128),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.length * 2, self.length * 4, 4,2,1), #16
			nn.BatchNorm3d(256),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.length * 4, self.length * 8, 4,2,1), #4
			nn.BatchNorm3d(512),
			nn.LeakyReLU(0.2), 

			nn.Conv3d(self.length * 8, 1, 4,2,0), #1
			nn.Sigmoid()						
			
			) 

	def forward(self, x):
		x = self.main(x)
		return x.view(-1,x.size(1))

if __name__ == '__main__':
	G = Generator(200,64, 11)
	D = Discriminator(11,64)
	z = Variable(torch.rand(16,200,1,1,1))
	X = G(z)
	D_X = D(X)
	print(X.shape, D_X.shape)


