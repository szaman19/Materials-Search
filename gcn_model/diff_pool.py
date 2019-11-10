import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
import pickle

import matplotlib.pyplot as plt

import numpy as np
import MOFDataset

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

max_nodes = 2000


class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 add_loop=False,
                 lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(3, 64, num_nodes, add_loop=True)
        self.gnn1_embed = GNN(3, 64, 64, add_loop=True, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# training_data_list = MOFDataset.MOFDataset(train=True).get_data()

	training_data_list = pickle.load(open('pickled_data.p','rb'))
	loader = DataLoader(training_data_list, batch_size = 1)

	# test_dl = MOFDataset.MOFDataset(train=False).get_data()
	test_dl = pickle.load(open('pickled_test_data.p','rb'))
	test_loader = DataLoader(test_dl, batch_size=1)

	model = Net().to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
	epoch = 20
	print("Starting Training:")
	print("*"*40)
	f
	or i in range(epoch):

		training_loss = 0
		for data in loader:
			data = data.to(device)
			optimizer.zero_grad()
			out = model(data)
			loss = criterion(out, torch.unsqueeze(data.y,1))
			# print(loss.item())
			training_loss += loss.item()
			loss.backward()
			optimizer.step()
		model.eval()
		total_loss = 0
		for test_data in test_loader:
			data = test_data.to(device)
			with torch.no_grad():
				pred= model(data)
			loss = criterion(pred, torch.unsqueeze(test_data.y,1))
			total_loss += loss.item()


		print("Epoch: ", i + 1, " Average Training MSE: ", training_loss / len(loader), " MSE for test is: ", total_loss / len(test_loader))


	print("*" * 40)
	print("Starting Test: ")
	print("*" * 40)



	model.eval()

	total_loss = 0

	vals = []
	for test_data in test_loader:
		data = test_data.to(device)
		with torch.no_grad():
			pred= model(data)
			vals.append((pred,torch.unsqueeze(test_data.y,1) ))
			# print(pred)
			# print(torch.unsqueeze(test_data.y,1))
		loss = criterion(pred, torch.unsqueeze(test_data.y,1))
		total_loss += loss.item()
	print("MSE for test is: ", total_loss / len(test_loader))

	vals.sort(key=lambda tup:tup[1])

	actuals = []
	pred = []

	# print(vals)

	log = open("vals2_1.log",'w')
	for each in vals:
		# print(each[0][0].item(), each[1][0].item())
		
		actuals.append(each[1][0])
		pred.append(each[0][0])

		log.write(str(each[0][0].item())+",")
		log.write(str(each[1][0].item())+"\n")

	log.close()
	indices = np.arange(len(actuals))

	plt.bar(indices, actuals, color="b", label="Actuals", )
	plt.bar(indices, pred, color="r", label="Predicted", alpha=0.5)
	axes = plt.gca()
	axes.set_ylim([0,16])
	plt.legend()
	plt.savefig("diff_pool.png", format="png")
	# plt.show()




	
if __name__ == '__main__':
	main()
