from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch 
from torch import Tensor 
from torch_geometric.nn.conv import MessagePassing

class MOLGCN(MessagePassing):
	"""docstring for MOLGCN"""
	def __init__(self, nn):
		super(MOLGCN, self).__init__()
		self.nn = nn

	def reset_parameters(self):
		self.nn.reset_parameters()
	
	def forward(self, x, edge_index, edge_attr):
		
		if isinstance(x, Tensor):
			x: PairTensor = (x, x)

		



	def message(self):

		