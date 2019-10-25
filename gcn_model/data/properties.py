import os 
import pandas as pd 
import glob 

def main():
	properties = pd.read_csv("2019-07-01-ASR-public_12020.csv")

	training_files = glob.glob("test/*.cif")
	
	training_files = [f.strip("test/").rstrip().strip(".cif") for f in training_files]
	counter = 0

	# print(training_files[2])
	for filenames in properties['filename']:
		if filenames in training_files:
			counter += 1 
		else:	
			properties = properties[ properties.filename != filenames]

	print(counter)
	properties.to_csv("properties.csv")
	
main()