import torch 
import os
from torch_geometric.data import Data
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element
import pandas as pd
import numpy as np
import torch_geometric.utils
import networkx as nx
import multiprocessing as mp 
import pickle

import warnings

import argparse

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

torch.multiprocessing.set_sharing_strategy('file_system')


warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_printoptions(profile="full")

class MOFDataset():
	"""docstring for MOFDataset"""
	def __init__(self,
		save_file_name, 
		train_dir = "/data/training",
		test_dir = "/data/test", 
		label = "LCD"):
		
		super(MOFDataset, self).__init__()

		self.train_dir = train_dir
		self.test_dir = test_dir
		self.save_file_name = save_file_name
		self.label_type = label

	def save_data(self):

		train_data = self.get_data(self.train_dir)
		
		print("Finished generating training data")
		train_file_name = "train_"+self.save_file_name + ".p"

		train_file = open(train_file_name, "wb")
		pickle.dump(train_data, train_file)

		train_file.close()



		test_data = self.get_data(self.test_dir)

		print("Finished generating test data")

		test_file_name = "test_"+self.save_file_name + ".p"

		test_file = open(test_file_name, "wb")
		pickle.dump(test_data, test_file)

		test_file.close()

	def get_data_helper(self, structure, y):

		structure = structure
		distance_matrix = structure.distance_matrix


		zero_indices = distance_matrix > 3
		
		distance_matrix[zero_indices] = 0


		graph = nx.from_numpy_matrix(distance_matrix.astype(np.double))
		num_nodes = distance_matrix.shape[0]
			# print(num_nodes)
		feature_matrix = self.get_feature_matrix(structure,num_nodes)
			
		data = torch_geometric.utils.from_networkx(graph)
		# data.x = torch.tensor(feature_matrix, dtype=torch.double)
		data.x = feature_matrix
		data.y = y
		# 
		# print(labels, " : ", num_nodes, " , ", y)
		return data		

	def get_data(self, data_dir):
		directory = os.getcwd() + data_dir +"/"

		labels = pd.read_csv(directory+"properties.csv")

		dataset = []

		size = len(labels['filename'])

		print(size)

		counters = list(range(size))

		directory_list = [directory + file +".cif" for file in labels['filename']]


		pool = mp.Pool(processes=40)

		structures = pool.map(self.cif_structure, directory_list)

		pool.close()


		labels = list(labels[self.label_type])
		print(len(structures))
		print(len(labels))


		print("structure parsing complete")
		# structures = [self.cif_structure(directory+file+".cif") for file in labels['filename']]

		data = zip(structures, labels)

		pool = mp.Pool(processes=40)
		dataset = pool.starmap(self.get_data_helper, data)

		pool.close()
		return dataset

	def cif_structure(self,file_name):
		parser = CifParser(file_name)
		structure = parser.get_structures()[0]
		# parse.close()
		return structure	

	def one_hot_encode(self, element):
		elements = ["H","N","C","O","Co","P","Zn","Ag","Cd","Cu","Fe"]
		return elements.index(element)

	def one_hot_test(self, val = 0, element="H"):
		one_hot = self.one_hot_encode(element)
		print(one_hot)

		true_val = np.zeros(11)
		true_val[val] = 1
		print(true_val)
		print(true_val == self.one_hot_encode(element))


	def get_feature_matrix(self, structure, num_nodes):
		feature_matrix = torch.zeros(num_nodes,11, dtype=torch.float)

		counter = 0
		for each in structure.sites:
			# vec = self.one_hot_encode(str(each.specie))
			# arr.append(vec)
			index = self.one_hot_encode(str(each.specie))
			feature_matrix[counter][index] = 1
			# element = Element(each.specie)
			# feature_matrix[counter][0] = element.atomic_radius_calculated
			# feature_matrix[counter][1] = element.atomic_mass
			# feature_matrix[counter][2] = element.row
			# feature_matrix[counter][3] = element.mendeleev_no

			counter +=1

		return feature_matrix




def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("save_file", help="Save file name")
	args = parser.parse_args()
	dataset = MOFDataset(args.save_file)
	dataset.save_data()

if __name__ == '__main__':
	main()

