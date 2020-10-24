import torch 
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import Set2Set
from torch_geometric.nn import GraphConv as GraphConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp
import pickle

import matplotlib.pyplot as plt

import numpy as np
import MOFDataset
import Net


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# training_data_list = MOFDataset.MOFDataset(train=True).get_data()
	# training_data_list = pickle.load(open('sparse_train_data_half_precision_sp.p','rb'))
	training_data_list = pickle.load(open('pickled_data.p','rb'))
	
	loader = DataLoader(training_data_list, batch_size = 64)

	# test_dl = MOFDataset.MOFDataset(train=False).get_data()
	# test_dl = pickle.load(open('sparse_test_data_half_precision_sp.p','rb'))
	# test_dl = pickle.load(open('pickled_test_data.p','rb'))
	
	# test_loader = DataLoader(test_dl, batch_size=256)

	model = Net.Net(13).to(device)
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
