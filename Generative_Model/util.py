import warnings 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Visualize_4DTensor(tensor, channels, threshold=1E-6, savefile="Visualize_4DTensor.png"):
	Channel_Titles = ["Energy Grid","H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"]
	if (len(tensor.shape) != 4):
		print("Tensor must be 4-dimensional. Tensor shape was: ", tensor.shape)
	else:
		fig = plt.figure()
		counter = 1 
		for channel in channels:
			ax = fig.add_subplot(len(channels), 1, counter, projection='3d')
			grid = tensor[channel]
			grid[grid < threshold] = 0
			ax.voxels(grid)
			ax.set_title(Channel_Titles[channel])
			counter +=1

		plt.legend()
		plt.savefig(savefile)


def Visualize_MOF(tensor, channels, threshold=1E-1, savefile="MOF.png"):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Channel_Titles = ["Energy Grid","H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"]

	for i,channel in enumerate(channels):
		grid = np.copy(tensor[channel])
		grid[grid < threshold] = 0
		ax.voxels(grid) 
	# plt.legend()
	plt.savefig(savefile)

def Visualize_MOF_Split(tensor, channels, threshold=1E-1, savefile="MOF.png"):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	Channel_Titles = ["Energy Grid","H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"]

	for i,channel in enumerate(channels):
		grid = np.copy(tensor[channel])
		grid[grid < threshold] = 0
		ax.voxels(grid) 
		plt.savefig(Channel_Titles[i]+"_"+savefile)
		# plt.close()