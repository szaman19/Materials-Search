import torch 
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
import MOFDataset

class Net(torch.nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super(Net, self).__init__()
		
		#channel in is  size of input features, channel out is 128
		self.conv1 = GraphConv(dataset.num_features, 128) 

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

def main():
	training_data_obj = MOFDataset.MOFDataset(train=True).get_data()
	# model = Net()
	# criterion = torch.nn.MSEloss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
	
if __name__ == '__main__':
	main()