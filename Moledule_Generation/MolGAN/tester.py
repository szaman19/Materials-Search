from model import Generator 
import numpy as np 
import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim 

from torch.autograd import Variable 
import torch_geometric.utils as graph_utils
from torch_geometric.data import Data

def post_process(inp):
	def listify(x):
		return x if type(x) == list or  type(x) == tuple else [x]
	def delistifiy(x):
		return x if len(x) > 1 else x[0]

	softmax = [F.softmax(e_logits, -1) for e_logits in listify(inp)]

	return [delistifiy(e) for e in (softmax)]


def main():
	Gen = Generator()
	Discrim = Discriminator()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	real_label = 1 
	fake_label  = 0

	criterion = nn.BCELoss()

	z = np.random.normal(0, 1, size=(1, 32))

	z = torch.from_numpy(z).to(device).float()

	adj, node = Gen(z)


	# print(adj)
	(adj_hat, node_hat) = post_process((adj, node))

	# print(adj_hat)
	# print(adj_hat.shape)

	adj_hat = adj_hat.squeeze()
	# print(adj_hat)
	# print(adj_hat.shape)

	index, value = graph_utils.dense_to_sparse(adj_hat)

	value = value.unsqueeze(1)
	print(node_hat.shape)
	print(index.shape)
	print(value.shape)

	data = Data()
	data.x = node_hat
	data.edge_index = index
	data.edge_attr = value
	data.y = fake_label

	










if __name__ == '__main__':
	main()
