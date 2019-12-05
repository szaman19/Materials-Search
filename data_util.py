import os 
import glob

import random

def main():

	file_name = os.getcwd() + "/data/files.log" 
	f = open(file_name,"r")
	file_names = [i for i in f.readlines()]

	source_dir = os.getcwd()+ "/data/structure_11660/"
	train_dir = os.getcwd()+"/gcn_model/data/training/"
	test_dir = os.getcwd()+"/gcn_model/data/test/"

	for cif_file in file_names:
		cif_file = cif_file.rstrip()
		num = random.random()
		if(num < .89):
			os.rename(source_dir+cif_file, train_dir+cif_file)
		else:
			os.rename(source_dir+cif_file, test_dir+cif_file)
main()
