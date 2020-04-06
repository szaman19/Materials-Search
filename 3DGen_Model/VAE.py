import numpy as np 
import torch 
import torch.autograd as autograd
import torch.nn as nn 
from torch.autograd import Variable 
import util

from MOFVAE import VAE_model 

cuda = True if torch.cuda.is_available() else False 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
	enc = VAE_model.Encoder(Num_features=12, Z_dim = 64).to(device)
	dec = VAE_model.Decoder(num_features = 12, z_dimension = 64, voxel_side_length=32).to(device)



	data = Variable(torch.rand(2,12,32,32,32)).to(device)

	optimizer = optim.Adam()
	z, mu, logvar = enc(data)

	print(z.shape, mu.shape, logvar.shape)

	X = z.view(z.size(0), z.size(1), 1,1,1)
	print(X.shape)

	decoded_z = dec(X)

	print(decoded_z[0].shape)

	chans = range(0,1)
	util.Visualize_4DTensor(decoded_z[0].cpu().detach().numpy(), chans)


		
