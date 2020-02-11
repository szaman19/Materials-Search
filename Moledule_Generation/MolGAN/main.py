from model import Generator 
from model import Discriminator 

import numpy as np 
import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 

from torch.autograd import Variable 
import torch_geometric.utils as graph_utils
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pickle
# from data import SmallWorld
def post_process(inp):
	def listify(x):
		return x if type(x) == list or  type(x) == tuple else [x]
	def delistifiy(x):
		return x if len(x) > 1 else x[0]

	softmax = [F.softmax(e_logits, -1) for e_logits in listify(inp)]

	return [delistifiy(e) for e in (softmax)]

def gen_samples(generator, device):
	fake_data = []
	fake_label = 0
	for i in range(32):
		z = np.random.normal(0,1,size=(1,32))
		z = torch.from_numpy(z).to(device).float()
		adj, node = generator(z)
		
		(adj_hat, node_hat) = post_process((adj, node))
		
		adj_hat = adj_hat.squeeze()
		
		index, value = graph_utils.dense_to_sparse(adj_hat)

		value = value.unsqueeze(1)

		data = Data(x=node_hat, edge_index=index, edge_attr=value, y=fake_label)

		# print(type(data))
		fake_data.append(data)

	return fake_data

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	Gen = Generator()
	D = Discriminator()
	# optimizerD = optim


	criterion = nn.BCELoss()

	real_label = 1 
	fake_label = 0
	
	optimizerD = optim.Adam(D.parameters(), lr=0.0001, betas=(0.9, 0.999))
	optimizerG = optim.Adam(Gen.parameters(), lr=0.01, betas=(0.9, 0.999))

	NUM_EPOCHS = 20

	for epoch in range(NUM_EPOCHS):
			##
			## First train the discriminator 
			##

		training_data_list = pickle.load(open('MOF_GENERATOR_32.p','rb'))
		fake_data = gen_samples(Gen, device)
		true_loader = DataLoader(training_data_list, batch_size=16)
		fake_loader = DataLoader(fake_data, batch_size=16) 

		real_loss = 0
		fake_loss = 0

		for data in true_loader:
			data = data.to(device)
			out = D(data)
			real_loss += -torch.mean(out)


		for data in fake_loader:
			data = data.to(device)
			out = D(data)
			fake_loss += torch.mean(out)
		disc_loss = fake_loss + real_loss

		optimizerD.zero_grad()
		disc_loss.backward()
		optimizerD.step()
		print("Discriminator Loss: ",disc_loss.item(), 
			"Real Loss:", real_loss.item(), " Fake Loss", fake_loss.item())
			##
			## Second train the generator
			##
		fake_data = gen_samples(Gen, device)
		fake_loader = DataLoader(fake_data, batch_size=16) 

		g_fake_loss  = 0
		for data in fake_loader:
			data = data.to(device)
			out = D(data)
			g_fake_loss += -torch.mean(out)
		
		optimizerG.zero_grad()
		g_fake_loss.backward()
		optimizerG.step()
		print("Generator Loss: ", g_fake_loss.item())
		print("*" * 40)
			##
			## Not really sure here
			##
if __name__ == '__main__':
	main()