import warnings 
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def Visualize_4DTensor(tensor, channels, threshold=1E-6):
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
		plt.show()

