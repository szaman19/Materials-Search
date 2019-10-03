import pandas as pd
import os

def main():
	os.chdir("../data")
	file_name = "/2019-07-01-ASR-public_12020.csv"
	file_name =os.getcwd() + file_name
	dataframe = pd.read_csv(file_name)
	print(dataframe.describe())

main()
