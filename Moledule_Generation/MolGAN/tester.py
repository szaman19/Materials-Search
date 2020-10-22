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
def post_process(inp):
	def listify(x):
		return x if type(x) == list or  type(x) == tuple else [x]
	def delistifiy(x):
		return x if len(x) > 1 else x[0]

	softmax = [F.softmax(e_logits, -1) for e_logits in listify(inp)]

	return [delistifiy(e) for e in (softmax)]


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	Discrim = Discriminator()
	Gen = Generator()

	gen_optim = optim.Adam(Gen.parameters(), lr=0.0001)
	discrim_optim = optim.Adam(Discrim.parameters(), lr= 0.0001)


	real_label = 1 
	fake_label  = 0

	criterion = nn.BCELoss()

	# z = np.random.normal(0, 1, size=(1, 32))
	# z = torch.from_numpy(z).to(device).float()
	# adj, node = Gen(z)
	# print(adj)
	# (adj_hat, node_hat) = post_process((adj, node))
	# print(adj_hat)
	# print(adj_hat.shape)
	# adj_hat = adj_hat.squeeze()
	# print(adj_hat)
	# print(adj_hat.shape)
	# index, value = graph_utils.dense_to_sparse(adj_hat)
	# value = value.unsqueeze(1)
	# print(node_hat.shape)
	# print(index.shape)
	# print(value.shape)
	# data = Data(x=x, edge_index=index, edge_attr=value, y=fake_label)

	training_data_list = pickle.load(open('MOF_GENERATOR_32.p','rb'))

	for epochs in range(10):
		fake_data = []
		for i in range(32):
			z = np.random.normal(0,1,size=(1,32))
			z = torch.from_numpy(z).to(device).float()
			adj, node = Gen(z)
			
			(adj_hat, node_hat) = post_process((adj, node))
			
			adj_hat = adj_hat.squeeze()
			
			index, value = graph_utils.dense_to_sparse(adj_hat)

			value = value.unsqueeze(1)

			data = Data(x=node_hat, edge_index=index, edge_attr=value, y=fake_label)

			# print(type(data))
			fake_data.append(data)

		true_loader = DataLoader(training_data_list, batch_size=16)
		fake_loader = DataLoader(fake_data, batch_size=16) 

		real_loss = 0
		fake_loss = 0

		for data in true_loader:
			data = data.to(device)
			# print(dir(data))
			# print((data.edge_attr).shape)
			# print((data.edge_index).shape)
			# print((data.weight).shape)
			out = Discrim(data)
			real_loss += -torch.mean(out)

		print("*" * 40)

		for data in fake_loader:
			data = data.to(device)
			# print(dir(data))
			# print((data.x).shape)
			# print((data.edge_attr).shape)
			# print((data.edge_index).shape)
			out = Discrim(data)
			fake_loss += torch.mean(out)
		disc_loss = fake_loss + real_loss

		discrim_optim.zero_grad()
		disc_loss.backward()
		discrim_optim.step()


		##
		## Train the generator
		##









if __name__ == '__main__':
	main()
