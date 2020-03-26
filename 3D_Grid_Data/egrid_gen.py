import glob 
import subprocess
import os 

import multiprocessing as mp



def gengrid(file_names, save_file_dir = "grid_data/training/"):
	counter = 1
	print("Hello")
	for files in file_names:
		file_name = files.split("/")[2]
		save_file_name = save_file_dir + file_name + ".grid"
		subprocess.run(["./egrid", files, save_file_name])
		print(counter, ":", len(file_names))
		counter +=1
	return "done"

def multi_processing(file_names, save_file_dir = "grid_data/test/" ):
	#print(file_names.split("/"))
	file_name = file_names.split("/")[2]
	save_file_name = save_file_dir + file_name + ".grid"
	out = subprocess.run(["./egrid",file_names, save_file_name])
	return out.returncode

def main():
	training_files = glob.glob("input_data/training/*.inp")
	test_files = glob.glob("input_data/test/*.inp")
	
	training_files = [(x, "grid_data/training/") for x in training_files]
	test_files = [(x, "grid_data/test/") for x in test_files]


	pool=  mp.Pool(processes=40)
	#results = [pool.apply(gengrid, args=(x, )) for x in file_lists]
	results = pool.starmap(multi_processing, test_files)
	print(results)
	
	pool=  mp.Pool(processes=40)
	results = pool.starmap(multi_processing, training_files)
	print(results)

main()
