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
	return out
def main():
	training_files = glob.glob("input_data/training/*.inp")
	test_files = glob.glob("input_data/test/*.inp")
	
	counter = 1
	n = int(len(training_files) // 40)
	file_lists = [training_files[i:i+n] for i in range(0, len(training_files), n)]

	tot = 0

	#print(type(file_lists[0]))
	#print(len(file_lists))
	pool=  mp.Pool(processes=41)
	#results = [pool.apply(gengrid, args=(x, )) for x in file_lists]
	results = pool.map(multi_processing, test_files)
	print(results)
	#print(results)
	#counter = 1
	
	'''
	for file in test_files:
		file_name = file[16:]
		save_file_name = "grid_data/test/"+file_name[:-4]+".grid"
		print(subprocess.run(["./egrid",file,save_file_name ]))
		print(counter, " : ", len(test_files))
		counter  +=1

	'''
main()
