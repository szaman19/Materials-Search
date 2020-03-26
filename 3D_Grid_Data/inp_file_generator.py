import glob 
import subprocess

def main():
	training_files = glob.glob("data/training/*.cif")
	test_files = glob.glob("data/test/*.cif")
	
	for file in training_files:
		file_name = file[14:]
		save_file_name = "input_data/training/"+file_name[:-4]+".inp"
		subprocess.run(["./cif2input",file_name,"data_ff_UFF",save_file_name ])

	for file in test_files:
		file_name = file[10:]
		save_file_name = "input_data/test/"+file_name[:-4]+".inp"
		subprocess.run(["./cif2input",file_name,"data_ff_UFF",save_file_name ])

main()
