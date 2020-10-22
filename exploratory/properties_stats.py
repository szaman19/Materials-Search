import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
	os.chdir("../data")
	file_name = "/2019-07-01-ASR-public_12020.csv"
	file_name =os.getcwd() + file_name
	dataframe = pd.read_csv(file_name)

	# for col in dataframe.columns:
	# 	print(col)
	print(dataframe[dataframe.columns[:5]].describe())
	# pd.DataFrame.hist(dataframe, column=["LCD","PLD"], grid=False, sharey=True)
	# plt.xlabel("LCD")
	# plt.ylabel("# Number of Structures")
	# plt.show()

main()
