import os.path as osp
import torch 
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

target = 0
class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

class Net(torch.nn.Module):
	"""docstring for Net"""
	def __init__(self, num_features):
		super(Net, self).__init__()
		
		#channel in is  size of input features, channel out is 128
		self.conv1 = GCNConv(num_features, 128) 

		#channel in is 128, channel out is 128. Ratio is 0.8
		self.pool1 = TopKPooling(128, ratio=0.8)
		
		#channel in is 128. channel out is 128
		self.conv2 = GCNConv(128,128)

		#channel in is 128, channel out is 128. Ratio is 0.8
		self.pool2 = TopKPooling(128, ratio=0.8)

		self.conv3 = GCNConv(128, 128)

		self.pool3 = TopKPooling(128, ratio=0.8)

		self.lin1 = torch.nn.Linear(256, 128)
		self.lin2 = torch.nn.Linear(128,256)
		self.lin3 = torch.nn.Linear(256, 1) #Continuous output

	def forward(self, data):
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
		x = F.dropout(x, p=1, training = self.training)
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		return x

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
	transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
	dataset = QM9(path, transform=MyTransform).shuffle()

	# Normalize targets to mean = 0 and std = 1.
	# mean = dataset.data.y[:, target].mean().item()
	# std = dataset.data.y[:, target].std().item()
	# dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

	# Split datasets.
	# test_dataset = dataset[:10000]
	# val_dataset = dataset[10000:20000]
	# train_dataset = dataset[20000:]
	
	# test_loader = DataLoader(test_dataset, batch_size=64)
	# val_loader = DataLoader(val_dataset, batch_size=64)
	# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


	# model = Net(dataset.num_features).to(device)

	# criterion = torch.nn.MSELoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
	# epoch = 200
	
	# print("Starting Training:")
	# print("*"*40)
	# for i in range(epoch):

	# 	training_loss = 0
	# 	for data in loader:
	# 		data = data.to(device)
	# 		optimizer.zero_grad()
	# 		out = model(data)
	# 		loss = criterion(out, torch.unsqueeze(data.y,1))
	# 		# print(loss.item())
	# 		training_loss += loss.item()
	# 		loss.backward()
	# 		optimizer.step()
	# 	print("Epoch: ", i + 1, " Average Training MSE: ", training_loss / len(loader))


	# print("*" * 40)
	# print("Starting Test: ")
	# print("*" * 40)


	# test_dl = MOFDataset.MOFDataset(train=False).get_data()

	# test_loader = DataLoader(test_dl, batch_size=3)

	# model.eval()

	# total_loss = 0
	# for test_data in test_loader:
	# 	data = test_data.to(device)
	# 	with torch.no_grad():
	# 		pred= model(data)
	# 		print(pred)
	# 		print(torch.unsqueeze(test_data.y,1))
	# 	loss = criterion(pred, torch.unsqueeze(test_data.y,1))
	# 	total_loss += loss.item()
	# print("MSE for test is: ", total_loss / len(test_loader))


	
if __name__ == '__main__':
	main()
