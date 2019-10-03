import pymatgen
import sys
from pymatgen.io.cif import CifParser
import os
import glob
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

'''
Given a valid cif filename, returns a pymatgen structure object

'''
def cif_structure(file_name):
	parser = CifParser(file_name)
	structure = parser.get_structures()[0]
	return structure

def main():
	os.chdir("../data/structure_11660/")
	files = glob.glob("*.cif")
	print(len(files))
	
	total_unique_species = set()

	file_write = open("species.dat","w")
	for file in files:
		structure = cif_structure(file)
		u_elements = num_species(structure)
		total_unique_species.union(u_elements)
		file_write.write(file)
		file_write.write(" ")
		file_write.write(str(u_elements))
		file_write.write("\n")
		print(u_elements)
	file_write.close()
	print("*"*80)

	print("Total number of unique elements in the dataset: ", len(total_unique_species))
	print(total_unique_species)


main()