import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import GraphConv as GraphConv
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
		self.conv1 = GraphConv(13, 32) 

		#channel in is 128, channel out is 128. Ratio is 0.5
		self.pool1 = TopKPooling(32, ratio=0.5)
		
		#channel in is 128. channel out is 128
		self.conv2 = GraphConv(32,32)

		#channel in is 128, channel out is 128. Ratio is 0.8
		self.pool2 = TopKPooling(32, ratio=0.8)

		self.conv3 = GraphConv(32, 32)

		self.pool3 = TopKPooling(32, ratio=0.8)
		
		self.conv4 = GraphConv(32, 32)

		self.pool4 = TopKPooling(32, ratio=0.8)


		self.lin1 = torch.nn.Linear(256, 128)
		self.bn1 = torch.nn.BatchNorm1d(num_features=128)
		self.lin2 = torch.nn.Linear(128,16)
		self.bn2 = torch.nn.BatchNorm1d(num_features=16)	
		self.lin3 = torch.nn.Linear(16, 1) #Continuous output

	def forward(self, data):
		x, edge_index, batch, weight = data.x, data.edge_index, data.batch, data.weight

	
		x = F.relu(self.conv1(x, edge_index, weight))
		x, edge_index, weight, batch, _, _ = self.pool1(x,edge_index,weight,batch)
		x1 = torch.cat([gmp(x,batch), gap(x,batch)],dim =1)

		x = F.relu(self.conv2(x, edge_index, weight))
		x, edge_index, weight, batch, _, _ = self.pool2(x, edge_index,weight, batch)
		x2 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

		x = F.relu(self.conv3(x, edge_index, weight))
		x, edge_index, weight, batch, _, _ = self.pool3(x, edge_index,weight , batch)
		x3 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

		x = F.relu(self.conv3(x, edge_index, weight))
		x, edge_index, _, batch, _, _ = self.pool4(x, edge_index,weight , batch)
		x4 = torch.cat([gmp(x,batch), gap(x,batch)], dim=1)

		x = torch.cat([x1, x2, x3, x4], dim=1)

		
		x = F.relu(self.lin1(x))
		x = F.dropout(x, p=.2, training = self.training)
		x = F.relu(self.lin2(x))
		#x = F.dropout(x, p=.5, training = self.training)
		x = self.lin3(x)
		return x



def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# training_data_list = MOFDataset.MOFDataset(train=True).get_data()

	training_data_list = pickle.load(open('radius_sparse_train_data_LCD.p','rb'))
	loader = DataLoader(training_data_list, batch_size = 64)

	# test_dl = MOFDataset.MOFDataset(train=False).get_data()
	test_dl = pickle.load(open('radius_sparse_test_data_LCD.p','rb'))
	test_loader = DataLoader(test_dl, batch_size=256)

	model = Net(13).to(device)
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=3E-4)
	epoch = 3000
	print("Starting Training:")
	print("*"*40)
	model.train()
	for i in range(epoch):
		model.train()
		# optimizer.zero_grad()
		training_loss = 0
		# count = 0
		for data in loader:
			data = data.to(device)
			optimizer.zero_grad()
			out = model(data)
			loss = criterion(out, torch.unsqueeze(data.y,1))
			# print(loss.item())
			training_loss += loss.item()
			#count +=1
			#print(training_loss)
			loss.backward()
			optimizer.step()
		
		model.eval()

		total_loss = 0
		# test_count = 0
		for test_data in test_loader:
			data = test_data.to(device)
			with torch.no_grad():
				pred= model(data)
			loss = criterion(pred, torch.unsqueeze(test_data.y,1))
			total_loss += loss.item()
			#test_count +=1
		# print("MSE for test is: ", total_loss / len(test_loader))

		print("Epoch: ", i + 1, " Average Training MSE: ", training_loss / len(loader), " Test MSE: ", total_loss / len(test_loader))


	print("*" * 40)
	print("Starting Test: ")
	print("*" * 40)


	# # test_dl = MOFDataset.MOFDataset(train=False).get_data()

	# # test_loader = DataLoader(test_dl, batch_size=1)

	model.eval()

	total_loss = 0

	vals = []
	test_loader = DataLoader(test_dl, batch_size=1)
	for test_data in test_loader:
		data = test_data.to(device)
		with torch.no_grad():
			pred= model(data)
			vals.append((pred,torch.unsqueeze(test_data.y,1)))
			# print(pred)
			# print(pred, torch.unsqueeze(test_data.y,1))
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
	plt.savefig("atom_species_lcd.png", format="png")
	# # plt.show()



def print_mode():
	model = Net(11)
	print(model)
	
if __name__ == '__main__':
	main()
	# print_mode()
