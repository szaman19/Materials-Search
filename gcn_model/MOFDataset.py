import torch 
import os
from torch_geometric.data import Data
from pymatgen.io.cif import CifParser
from pymatgen.core.periodic_table import Element
import pandas as pd
import numpy as np
import torch_geometric.utils
import networkx as nx
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_printoptions(profile="full")

class MOFDataset():
	"""docstring for MOFDataset"""
	def __init__(self, 
		train = True):
		super(MOFDataset, self).__init__()

		if train:
			self.data_dir = "/data/training/"
		else:
			self.data_dir = "/data/test/"

	def get_data(self):
		""" returns a list of Data objects """
		# print(type(directory)) 
		# files = glob.glob(self.data_dir+"*.cif")
		directory = os.getcwd() + self.data_dir

		labels = pd.read_csv(directory+"properties.csv")
		# counter = 1


		dataset = []

		counter = 0


		size = len(labels['filename'])
		# size = 300
		# # print(size)

		# for i in range(0,size, 40):
		# 	num_processes  = 0
		# 	if (size > i + 40):
		# 		num_processes = 40
		# 	else:
		# 		num_processes = size - i

		# 	pool = Pool(processes=num_processes)
		# 	results = [pool.apply_async(self.get_data_helper, args=(labels,i+x)) for x in range(num_processes)]
		# 	out = [p.get() for p in results]

		# 	for each in out:
		# 		dataset.append(each)
		# 	pool.close()
		# 	pool.join()
		# 	print("Data Loaded: ", i+num_processes,"/",size)

			# print(out) 





		# print(feature_matrix)
		# size = len(labels['filename'])

		# with Pool(processes=20) as pool:
		# 	dataset = []
		# 	for i in range(20):
		# 		resuls = pool.map(self.get_data_helper, args=(labels,arr[i],arr[i] + steps, size, )) 
		# 		vals = resuls.get()
		# 		for each in vals:
		# 			dataset.append(each)
		# print(dataset)
		for file in labels['filename']:
			if(os.path.exists(directory+file+".cif")):

				file  = labels['filename'][counter]
				structure = self.cif_structure(directory+file+".cif")
				distance_matrix = structure.distance_matrix

				num_nodes = distance_matrix.shape[0]
				if (num_nodes < 2000):

					# distance_matrix = ((distance_matrix == 0)*1000) + distance_matrix 
					# distance_matrix = 1 / distance_matrix
					distance_matrix = (distance_matrix < 3) * distance_matrix
					graph = nx.from_numpy_matrix(distance_matrix.astype(np.double))

					feature_matrix = self.get_feature_matrix(structure, num_nodes)
							
					data = torch_geometric.utils.from_networkx(graph)
					data.x = feature_matrix
						# data.x = torch.ones(num_nodes,1)
							# data.x = feature_matrix
					data.y = labels['PLD'][counter]
						# print(file, num_nodes, labels['LCD'][counter])
							
					dataset.append(data)
					
					print("Elements loaded: ",counter, "/", size)

				counter +=1
				# if counter == 3:
				# 	break
			else:
				print("Not ok skipping: ", file)
		return dataset

	def cif_structure(self,file_name):
		parser = CifParser(file_name)
		structure = parser.get_structures()[0]
		# parse.close()
		return structure	

	def one_hot_encode(self, element):
		elements = ["H","N","C","O","Co","P","Zn","Ag","Cd","Cu","Fe"]
		# one_hot_vector = torch.zeros(1,11)
		# one_hot_vector = np.zeros(len(elements))
		# one_hot_vector[elements.index(element)] = 1
		return elements.index(element)

	def one_hot_test(self, val = 0, element="H"):
		one_hot = self.one_hot_encode(element)
		print(one_hot)

		true_val = np.zeros(11)
		true_val[val] = 1
		print(true_val)
		print(true_val == self.one_hot_encode(element))


	def get_feature_matrix(self, structure, num_nodes):
		feature_matrix = torch.zeros(num_nodes,13, dtype=torch.float)

		counter = 0
		for each in structure.sites:
			# vec = self.one_hot_encode(str(each.specie))
			# arr.append(vec)
			index = self.one_hot_encode(str(each.specie))
			feature_matrix[counter][index] = 1
			element = Element(each.specie)
			feature_matrix[counter][11] = element.atomic_radius_calculated
			feature_matrix[counter][12] = element.atomic_mass
			counter +=1

		return feature_matrix

	def get_data_helper(self, labels, counter):
		x = counter
		directory = os.getcwd() + self.data_dir

		file  = labels['filename'][x]
		structure = self.cif_structure(directory+file+".cif")
		distance_matrix = structure.distance_matrix

		graph = nx.from_numpy_matrix(distance_matrix.astype(np.double))
		num_nodes = distance_matrix.shape[0]
			# print(num_nodes)
		feature_matrix = self.get_feature_matrix(structure,num_nodes)
			
		data = torch_geometric.utils.from_networkx(graph)
			# data.x = torch.tensor(feature_matrix, dtype=torch.double)
		data.x = feature_matrix
		data.y = labels['LCD'][x]
		# 

		return data




