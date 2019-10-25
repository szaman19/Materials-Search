import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

import MOFDataset

class Net(torch.nn.Module):
	"""docstring for Net"""
	def __init__(self, num_features):
		super(Net, self).__init__()
		
		#channel in is  size of input features, channel out is 128
		self.conv1 = GraphConv(num_features, 128) 

		#channel in is 128, channel out is 128. Ratio is 0.8
		self.pool1 = TopKPooling(128, ratio=0.8)
		
		#channel in is 128. channel out is 128
		self.conv2 = GraphConv(128,128)

		#channel in is 128, channel out is 128. Ratio is 0.8
		self.pool2 = TopKPooling(128, ratio=0.8)

		self.conv3 = GraphConv(128, 128)

		self.pool3 = TopKPooling(128, ratio=0.8)

		self.lin1 = torch.nn.Linear(256, 128)
		self.lin2 = torch.nn.Linear(128,64)
		self.lin3 = torch.nn.Linear(64, 1) #Continuous output

	def forward(self, forward):
		x, edge_index, batch = data.x, data.edge_index, data.batch

	
		x = F.relu(self.conv1(x, edge_index))
		x, edge_index, _, batch, _, _ = self.pool1(x,edge_index,None,batch)
		x1 = torch.cat([gmp(x,batch), gap(x,batch)],dim =1)

		x = F.relu(self.conv2(x, edge_index))
		x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
		x2 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

		x = F.relu(self.conv3(x, edge_index))
		x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
		x3 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

		x = x1 + x2 + x3 

		
		x = F.relu(self.lin1(x))
		x = F.dropout(x, p=0.5, training = self.training)
		x = F.relu(self.lin2(x))
		x = self.lin3(x)

def main():
	training_data_list = MOFDataset.MOFDataset(train=True).get_data()

	loader = DataLoader(training_data_list, batch_size = 16, shuffle = True)

	print(len(loader))
	# training_data_obj.one_hot_test(4,"Co")
	# model = Net(11)
	# criterion = torch.nn.MSELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
	
if __name__ == '__main__':
	main()