import torch 
import os
from torch_geometric.data import Data
import pandas as pd


class MOFDataset():
	"""docstring for MOFDataset"""
	def __init__(self, 
		train = True):
		super(MOFDataset, self).__init__()

		if train:
			self.data_dir = "/data/train/"
		else:
			self.data_dir = "/data/test/"

	def get_data(self):
		""" returns a list of Data objects """
		print(os.getcwd() + self.data_dir)
		files = glob.glob("*.cif")

		prop = "property.csv"
		

