import dgl 
import torch as tch 
import networkx as nx
import os
import glob
import pymatgen
import sys
from pymatgen.io.cif import CifParser

def cif_structure(file_name):
	parser = CifParser(file_name)
	structure = parser.get_structures()[0]
	return structure

def main():
	os.chdir("../data/structure_11660/")
	files = glob.glob("AHOKOX*.cif")
	structure = cif_structure(files[0])
	distance_matrix = structure.distance_matrix
	graph = nx.from_numpy_matrix(distance_matrix)
	dg_graph = dgl.DGLGraph()
	dg_graph.from_networkx(graph)
	
	print(type(dg_graph))


if __name__ == '__main__':
	main()
