import glob 
import subprocess
import multiprocessing as mp

def train_files_gen(file):
	file_name = file[14:]
	save_file_name = "input_data/training/"+file_name[:-4]+".inp"
	ret = subprocess.run(["./cif2input","data/training/"+file_name,"data_ff_UFF",save_file_name ])
	#print(ret)
	return ret.returncode

def test_files_gen(file):
	file_name = file[10:]
	save_file_name = "input_data/test/"+file_name[:-4]+".inp"
	ret = subprocess.run(["./cif2input","data/test/"+file_name,"data_ff_UFF",save_file_name ])
	return ret.returncode 

def main():
	training_files = glob.glob("data/training/*.cif")
	test_files = glob.glob("data/test/*.cif")
	
	pool = mp.Pool(processes=40)
	results = pool.map(train_files_gen, training_files)
	print(results)

	pool = mp.Pool(processes=40)
	results = pool.map(test_files_gen, test_files)
	print(results)
main()
