import pymatgen
import sys
from pymatgen.io.cif import CifParser
from multiprocessing import Pool
import os
import math
import glob
import networkx as nx
import numpy as np
class Periodic_Struture(object):
	"""docstring for Periodic Struture"""
	def __init__(self, file_name):
		super(Periodic_Struture, self).__init__()
		self.distance_matrix = self.cif_distance_matrix(file_name)
		self.file_name = file_name
		self.r = np.amax(self.distance_matrix)

	def cif_distance_matrix(self,file_name):
		parser = CifParser(file_name)
		structure = parser.get_structures()[0]
		temp = structure.distance_matrix

		return temp
	def get_min_radius(self):

		self.min_radius_helper(self.r)

	def min_radius_helper(self,radius):


		# print("Radius: ", radius, " Connected: ", is_connected)

		new_radius = radius / 2
		prev_radius = radius

		is_connected = False
		while(not(abs(new_radius - self.r) < .1)):
			temp = (self.distance_matrix < new_radius) * self.distance_matrix
			graph = nx.from_numpy_array(temp)
			is_connected = nx.is_connected(graph)

			# print("Radius: ", new_radius, "Is Connected: ",is_connected)
			# print(new_radius, ", ", self.r)
			if(is_connected):
				self.r= new_radius
				new_radius = new_radius / 2
			else:
				new_radius = new_radius + (self.r - new_radius) / 2
			# prev_radius = new_radius

		# self.r = prev_radius

def func(files):

	return_val = []
	for f in files:
		struct = Periodic_Struture(f)
		struct.get_min_radius()
		return_val.append(str((f, str(struct.r))))
	return return_val


def main():
	os.chdir("../data/structure_11660/")
	files = glob.glob("*.cif")

	# struct = Periodic_Struture(file)
	# print(struct.r)
	# struct.get_min_radius()

	Num_Processes = 20

	num_files = len(files)

	file_chunks = [ files[int((num_files/Num_Processes) * i): int((num_files / Num_Processes * (i+1)))] for i in range(Num_Processes)]

	for each in file_chunks:
		print(each)
	pool = Pool(processes=Num_Processes)
	results = [pool.apply_async(func, args=(file_chunks[i],)) for i in range(Num_Processes)]
	output = [p.get() for p in results]

	log = open("min_radius_2.log","w")

	for returned_list in output:
		for each in returned_list:
			log.write(each)
			log.write("\n")
	log.close()

	# print(struct.r)




if __name__ == '__main__':
	main()

