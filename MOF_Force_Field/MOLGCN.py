from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch 
from torch import Tensor 
from torch_geometric.nn.conv import MessagePassing

class MOLGCN(MessagePassing):
	"""docstring for MOLGCN"""
	def __init__(self,
				 nn, 
				 aggr = 'add',
				 **kwargs):
		super(MOLGCN, self).__init__()
		self.nn = nn
		self.aggr = aggr

	def reset_parameters(self):
		self.nn.reset_parameters()
	
	def forward(self, x, edge_index, edge_attr, size = None):
		
		if isinstance(x, Tensor):
			x: PairTensor = (x, x)

		out = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size)

		return out 


	def message(self, x_i, x_j, edge_attr):
		
		z = torch.cat([x_i, x_j, edge_attr], dim = -1)
		print(z)
		return self.nn(z)


		