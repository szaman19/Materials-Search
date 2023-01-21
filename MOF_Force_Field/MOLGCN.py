from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch 
from torch import Tensor 
from torch_geometric.nn.conv import MessagePassing
import torch.nn as neural_net
import torch.nn.functional as F

class MOLGCN(MessagePassing):
	"""docstring for MOLGCN"""
	def __init__(self,
				 nn, 
				 aggr = 'add',
				 learn_input = False,
				 feature_size = 4,
				 **kwargs):
		super(MOLGCN, self).__init__()
		self.nn = nn
		self.aggr = aggr
		self.learn_input = learn_input

		self.bond_representation_learner = None



	def reset_parameters(self):
		self.nn.reset_parameters()
	
	def forward(self, x, edge_index, edge_attr, size = None):
		
		if isinstance(x, Tensor):
			x: PairTensor = (x, x)

		out = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size)

		return out 


	def message(self, x_i, x_j, edge_attr):

		bonds = x_i + x_j
		z = torch.cat([bonds, edge_attr], dim = -1)
		return self.nn(z)


		