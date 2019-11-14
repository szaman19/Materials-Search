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

class Net(torch.nn.Module):
	"""docstring for Net"""
	def __init__(self, num_features):
		super(Net, self).__init__()
		
		#channel in is  size of input features, channel out is 128
		self.conv1 = GraphConv(11, 128) 

		#channel in is 128, channel out is 128. Ratio is 0.5
		self.pool1 = TopKPooling(128, ratio=0.5)
		
		#channel in is 128. channel out is 128
		self.conv2 = GraphConv(128,128)

		#channel in is 128, channel out is 128. Ratio is 0.8
		self.pool2 = TopKPooling(128, ratio=0.8)

		self.conv3 = GraphConv(128, 128)

		self.pool3 = TopKPooling(128, ratio=0.8)

		self.lin1 = torch.nn.Linear(256, 512)
		self.lin2 = torch.nn.Linear(512,64)
		self.lin3 = torch.nn.Linear(64, 1) #Continuous output

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
		x = F.dropout(x, p=.7, training = self.training)
		x = F.relu(self.lin2(x))
		x = self.lin3(x)
		return x



def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# training_data_list = MOFDataset.MOFDataset(train=True).get_data()

	training_data_list = pickle.load(open('sparse_train_data_half_precision_sp.p','rb'))
	loader = DataLoader(training_data_list, batch_size = 2)


	# test_dl = MOFDataset.MOFDataset(train=False).get_data()



	test_dl = pickle.load(open('sparse_test_data_half_precision.p','rb'))
	test_loader = DataLoader(test_dl, batch_size=1)

	model = Net(11).to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=5E-4)
	epoch = 100
	print("Starting Training:")
	print("*"*40)
	for i in range(epoch):
		model.train()
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
		# print("MSE for test is: ", total_loss / len(test_loader))

		print("Epoch: ", i + 1, " Average Training MSE: ", training_loss / len(loader), "Avg Test MSE: ", total_loss / len(test_loader))


	print("*" * 40)
	print("Starting Test: ")
	print("*" * 40)


	# test_dl = MOFDataset.MOFDataset(train=False).get_data()

	# test_loader = DataLoader(test_dl, batch_size=1)

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

	log = open("vals2_2.log",'w')
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
	plt.savefig("atom_species_p7_2_SPARSE.png", format="png")
	# plt.show()



def print_mode():
	model = Net(11)
	print(model)
	
if __name__ == '__main__':
	main()
	# print_mode()
