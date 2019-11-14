import pymatgen
import sys
from pymatgen.io.cif import CifParser
from pymatgen.core.lattice import Lattice
import os
import math

def cif_structure(file_name):
	parser = CifParser(file_name)
	structure = parser.get_structures()[0]
	# print(structure[0].distance_matrix)
	# print(type(structure[0].distance_matrix))
	
	temp = structure.distance_matrix

	# temp = (temp < 2.5) * tempz

	# print(temp) z
	# counter = 0
	# for each in structure.sites:
	# 	counter +=1
	# 	print(each.species,distance(each.coords[0], each.coords[1], each.coords[2]))
	
	# for i in range(counter):
	# 	print(structure.distance_matrix[0][i])
	

	return temp


def distance(x,y,z):
	return math.sqrt(x**2 + y**2 + z**2)
def main():
	os.chdir("../data/")
	file = "AHOKOX_clean.cif"

	distance_matrix = cif_structure(file)
	
	# print(distance_matrix)
	# cif_lattice(file)




main()