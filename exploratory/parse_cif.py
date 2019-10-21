import pymatgen
import sys
from pymatgen.io.cif import CifParser
import os
import glob
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
Given a structure, find the number of unique elements present
and return a list of elements
'''
def num_species(structure):

	sites = structure.as_dict()['sites']
	species = set()
	for site in sites:
		elements = site['species']
		for each in elements:
			species.add(each['element'])
	return species

def num_elements(structure):
	sites = structure.as_dict()['sites']
	return len(sites)
'''
Given a valid cif filename, returns a pymatgen structure object

'''
def cif_structure(file_name):
	parser = CifParser(file_name)
	structure = parser.get_structures()[0]
	return structure

def gen_3d_Plot(structure):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	dims = (25,10,5)
	numpy_pos = np.zeros(dims, dtype=bool) 
	numpy_color = np.zeros(dims, dtype=str)

	dic_color = {"Cu":"yellow","O":"blue","C":"green","P":"red"}
	for each in structure.sites:
		# print(type(each.specie),type(each.coords))
		new_coords = [0]*3
		for i  in range(len(each.coords)):
			new_coords[i] =  int(-1* each.coords[i])
		x = new_coords[0]
		y = new_coords[1]
		z = new_coords[2]
		numpy_pos[x][y][z] = True
		numpy_color[x][y][z] = dic_color[str(each.specie)]

	ax.voxels(numpy_pos,facecolors=numpy_color, edgecolor="k")
	plt.show()
def smaller(x1, x2):
	return int((x1 <= x2))*x1 + int((x1 > x2)) *x2
def bigger(x1, x2):
	return int((x1 >= x2))*x1 + int((x1 < x2)) *x2

def min_max_coords(structure):
	site = structure.sites[0]
	coord = site.coords

	x_min = coord[0]
	x_max = coord[0] 
	y_min = coord[1]
	y_max = coord[1]
	z_min = coord[2]
	z_max = coord[2]

	for each in structure.sites:
		coord = each.coords

		x_min = smaller(x_min,coord[0])
		y_min = smaller(y_min,coord[1])
		z_min = smaller(z_min,coord[2])

		x_max = bigger(x_max,coord[0])
		y_max = bigger(y_max,coord[1])
		z_max = bigger(z_max,coord[2])
	return x_min,x_max,y_min,y_max,z_min, z_max


def min_max_dataset(dataset):
	structure = cif_structure(dataset[0])
	x_min,x_max,y_min,y_max,z_min, z_max = min_max_coords(structure)

	for file in dataset:
		structure = cif_structure(file)
		x_min_s,x_max_s,y_min_s,y_max_s,z_min_s, z_max_s = min_max_coords(structure)
		x_min = smaller(x_min,x_min_s)
		y_min = smaller(y_min,y_min_s)
		z_min = smaller(z_min,z_min_s)

		x_max = bigger(x_max,x_max_s)
		y_max = bigger(y_max,y_max_s)
		z_max = bigger(z_max,z_max_s)
	return x_min,x_max,y_min,y_max,z_min, z_max
def main():
	os.chdir("../data/structure_11660/")
	files = glob.glob("*.cif")
	print(len(files))
	

	# structure = cif_structure(files[0])
	# gen_3d_Plot(structure)
	# x_min= x_max=y_min=y_max=z_min= z_max = 0
	# x_min,x_max,y_min,y_max,z_min, z_max = min_max_dataset(files)

	# print(x_min,x_max,y_min,y_max,z_min, z_max)
	# structure = cif_structure(files[10])
	# distance_matrix = structure.distance_matrix
	# print(files[0])
	# print(distance_matrix.shape)
	# graph = nx.from_numpy_matrix(distance_matrix)
	
	# nx.draw(graph)
	# plt.show()
	# total_unique_species = set()

	# file_write = open("sizes.dat","w")
	elements = {}
	for file in files:
		structure = cif_structure(file)
		u_elements = num_species(structure)
		# print(u_elements)
		for each in u_elements:
			if(each in elements):
				elements[each] += 1
			else:
				elements[each] = 1

	for each in elements.keys():
		print(each, elements[each])

	# 	# total_unique_species.union(u_elements)
	# 	file_write.write(file)
	# 	file_write.write(" ")
	# 	file_write.write(str(u_elements))
	# 	file_write.write("\n")
	# 	print(u_elements)
	# file_write.close()
	# print("*"*80)

	# print("Total number of unique elements in the dataset: ", len(total_unique_species))
	# print(total_unique_species)


main()