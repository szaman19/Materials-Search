import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import GraphConv  
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp


class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, input_vector_dim = 32, num_nodes=32, num_features=4, num_edge_features = 1):
		super(Generator, self).__init__()

		#Z vector to generate continuous graph from. Default is 32 
		self.z_dim = input_vector_dim
		self.N = num_nodes # Number of atoms 
		self.T = num_features # Types of atoms 
		self.Y = num_edge_features # Edge features 

		#Map the Z vector to a higher dimensional vector
		self.layer_1 = nn.Linear(self.z_dim, 128)
		self.layer_1_act = nn.Tanh()
		self.layer_2 = nn.Linear(128,256)
		self.layer_2_act = nn.Tanh()
		self.layer_3 = nn.Linear(256,512)
		self.layer_3_act = nn.Tanh()

		# This should match the dimensions for the A tensor (Adjacency Tensor) 
		self.edges_layer = nn.Linear(512, self.N * self.N * self.Y)

		# This should match the dimensions for the X matrix (Feature Matrix)
		self.node_layer = nn.Linear(512, self.N * self.T)


	def forward(self, x):
		output = self.layer_1(x)
		output = self.layer_1_act(output)
		output = self.layer_2(output)
		output = self.layer_2_act(output)
		output = self.layer_3(output)
		output = self.layer_3_act(output)

		adj_mat = self.edges_layer(output).view(-1, self.Y, self.N, self.N)

		#Make tensor symmetric in H W dimensions
		adj_mat = (adj_mat + adj_mat.permute(0,1,3,2))/2
		#Move the edge attribute to the end 
		adj_mat = adj_mat.permute(0,2,3,1)

		feat_mat = self.node_layer(output).view(-1, self.N, self.T)

		return adj_mat, feat_mat 


class Discriminator(object):
	"""docstring for Discriminator"""
	def __init__(self, num_input_features = 4):
		super(Discriminator, self).__init__()
		self.atom_types = num_input_features

		self.conv1 = GraphConv(self.atom_types, 128)
		self.pool1 = TopKPooling(128, ratio=0.8)

		self.conv2 = GraphConv(128, 128)
		self.pool2 = TopKPooling(128, ratio = 0.8)


		self.lin1 = nn.Linear(128, 64)
		self.lin2 = nn.Linear(64,16)
		self.lin3 = nn.Linear(16,1)

	def forward(self, data):
		x, edge_index, batch, weight = data.x, data.edge_index, data.batch, data.weight

		x = F.relu (self.conv1(x,edge_index, weight))
		x, edge_index, weight, batch, _,_ = self.pool1(x, edge_index, weight, batch)
		x1 = torch.cat([gmp(x,batch),  gap(x,batch)], dim=1)

		x = F.relu (self.conv1(x,edge_index, weight))
		x, edge_index, weight, batch, _,_ = self.pool2(x, edge_index, weight, batch)
		x2 = torch.cat([gmp(x,batch),  gap(x,batch)], dim=1)

		x = torch.cat([x1,x2], dim=1)
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x = F.relu(self.lin3(x))

		return x



		
		

