from pymatgen.io.cif import CifParser
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter

class CIFtoTensor(object):
	"""docstring for CIFtoTensor"""
	def __init__(self):
		super(CIFtoTensor, self).__init__()
		self.default_atoms = ["H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"]
		

	def get_cif_file(file_name = "AHOKOX_clean.cif"):
		cif_file = CifParser(file_name)
		return cif_file
	def get_pymat_struct(cif_file):
		struct = cif_file.get_structures()
		if(len(struct) > 1):
			warnings.warn("Multiple Structures generated")
		struct = struct[0]
		return struct

	def get_image_distance(cif_file):
		struct = cif_file.get_structures()

		if(len(struct) > 1):
			warnings.warn("Multiple Structures generated")
		struct = struct[0]
		return struct.distance_matrix

	def to3DTensor(pymat_struc ,atom_species=[], dimensions = (32,32,32), gaussian_blurring=True, half_precision = False):
		

		if(gaussian_blurring):
			warnings.warn("Haven't implemented blurring. Returning tensor with pointwise values")

		if(half_precision):
			warnings.warn("Using half precision tensors. May not be compatible with training on network without some rewrites")
		'''
		 TO DO: 

		 Possibly a useful idea to just integer values for the representation. Look into it later. 
		'''
		if(len(atom_species) ==0):
			# print('Atom Species not specified. Using default_atoms:  ["H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"] ')
			atom_species = ["H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"]

		dimensions = (len(atom_species),dimensions[0],dimensions[1], dimensions[2])
		mol_tensor = np.zeros(dimensions)

		MAX = 31
		NORMALIZE = 1
		shifted = False

		site_0 = None
		for site in pymat_struc.sites:
			if(site.x ==0 and site.y == 0 and site.z == 0): 
				site_0 = site
			else:
				specie = atom_species.index(str(site.specie))
				x = 0 
				y = 0 
				z = 0
				if(site.x < 0):
					shifted = True
					x = MAX - (abs(site.x) / NORMALIZE)
				if(site.x < 0):
					shifted = True
					y = MAX - (abs(site.y) / NORMALIZE)
				if(site.x < 0):
					shifted = True
					z = MAX - (abs(site.z) / NORMALIZE)
				assert x >= 0 and  x <= MAX
				assert y >= 0 and y <= MAX
				assert z >= 0 and z <= MAX

				mol_tensor[specie] =add_mol_gaussian(mol_tensor,specie,x,y,z)
		site_0_specie = atom_species.index(str(site_0.specie))
		if(shifted):
			mol_tensor[site_0_specie] = add_mol_gaussian(mol_tensor, site_0_specie, MAX,MAX,MAX)
		else:
			mol_tensor[site_0_specie] = add_mol_gaussian(mol_tensor, site_0_specie, 0,0,0)

		for i in range(len(atom_species)):
			mol_tensor[i] = gaussian_filter(mol_tensor[i], sigma = 0.5)

		return mol_tensor
def add_mol_disrete(tensor, specie, x, y, z):
	tensor[specie][x][y][z] += 1
	return tensor[specie]

def add_mol_gaussian(tensor, specie, x,y,z, variance=0.5):
	shape = tensor[specie].shape
	distances = np.zeros(shape)

	for x_i in range(shape[0]):
		for y_i in range(shape[1]):
			for z_i in range(shape[2]):
				distances[x_i][y_i][z_i] = (-0.5)*((x_i - x)**2 + (y_i - y)**2 +(z_i-z)**2)/(variance**2)
	distances = np.exp(distances)
	distances = np.power(2/np.pi,3/2) * distances
	assert distances.shape == shape 
	tensor[specie] += distances
	
	return tensor[specie]
def Plot3D(tensor):
	fig = plt.figure(figsize = plt.figaspect(0.25))
	dims = (32,32,32)
	ax1 = fig.add_subplot(1,4,1,projection='3d')
	ax2 = fig.add_subplot(1,4,2,projection='3d')
	ax3 = fig.add_subplot(1,4,3,projection='3d')
	ax4 = fig.add_subplot(1,4,4,projection='3d')

	
	
	# for i in range(11):
		# ax.voxels(tensor[i], edgecolor="k")
	ax1.voxels(tensor[1], edgecolor="k", facecolor="blue")
	ax2.voxels(tensor[5], edgecolor="k", facecolor = "orange")
	ax3.voxels(tensor[3], edgecolor="k", facecolor="yellow")
	ax4.voxels(tensor[4], edgecolor="k", facecolor="red")

	ax1.set_title("Oxygen")
	ax2.set_title("Copper")
	ax3.set_title("Carbon")
	ax4.set_title("Phosphorus")

	plt.legend()
	plt.show()


def main():
	cif_file = CIFtoTensor.get_cif_file()
	struc = CIFtoTensor.get_pymat_struct(cif_file)
	mol_tensor = CIFtoTensor.to3DTensor(struc)
	Plot3D(mol_tensor)


	
if __name__ == '__main__':
	main()
