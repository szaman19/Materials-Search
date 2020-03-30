import pickle 
import Voxel_MOF
import glob 
import multiprocessing as mp 

def get_MOF_data(inp_file,grid_file):
	MOF = Voxel_MOF.Voxel_MOF(inp_file,grid_file)
	return MOF

def main():
	training_files_inp = glob.glob("input_data/training/*.inp")
	training_files_grid = glob.glob("grid_data/training/*.grid")
	tf = zip(training_files_inp, training_files_grid)

	pool = mp.Pool(processes=40)

	results = pool.starmap(get_MOF_data,tf)

	training_pickle_file = open("Training_MOFS.p","wb")

	pickle.dump(results, training_pickle_file)

	test_files_inp = glob.glob("input_data/test/*.inp")
	test_files_grid = glob.glob("grid_data/test/*.grid")
	tf = zip(test_files_inp, test_files_grid)

	pool = mp.Pool(processes=40)

	results = pool.starmap(get_MOF_data,tf)

	test_pickle_file = open("Test_MOFS.p","wb")

	pickle.dump(results, test_pickle_file)






main()
