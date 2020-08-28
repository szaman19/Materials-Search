from pymatgen.io.cif import CifParser
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings

class CIFtoVoxel(object):
	"""docstring for CIFtoVoxel"""
	def __init__(self, filename):
		super(CIFtoVoxel, self).__init__()
		self.filename = filename
		cif_file = CifParser(self.filename)

		self.struct = cif_file.get_structures()[0]

		self.lattice = self.struct.lattice

		self.voxel = self.__generate_voxel()


	def __generate_voxel(self):
		return self.to3DTensor()


	def to3DTensor(self ,
		atom_species=[], 
		dimensions = (32,32,32), 
		gaussian_blurring=True, 
		spread = 0.1):
		

		# if(half_precision):
		# 	warnings.warn("Using half precision tensors. \
		# 		May not be compatible with training on \
				# network without some rewrites")
		'''
		 TO DO: 

		 Possibly a useful idea to just integer values for the representation. 
		 Look into it later. 
		'''
		if(len(atom_species) ==0):
			# print('Atom Species not specified. Using default_atoms: 
			#["H","O", "N", "C", "P", "Cu","Co","Ag","Zn","Cd", "Fe"] ')
			atom_species = ["H","O", "N", "C", "P",
			 "Cu","Co","Ag","Zn","Cd", "Fe"]

		dimensions = (len(atom_species),dimensions[0],
				      dimensions[1], dimensions[2])
		mol_tensor = np.zeros(dimensions)

		MAX = dimensions[1]-1
		NORMALIZE_A = self.lattice.a 
		NORMALIZE_B = self.lattice.b
		NORMALIZE_C = self.lattice.c

		site_0 = None
		for site in self.struct.sites:
			x = site.a
			y = site.b
			z = site.c
			specie = atom_species.index(str(site.specie))
			# print(x,y,z)
			mol_tensor[specie] = self.add_mol_gaussian(mol_tensor,
												  specie,
												  x,
												  y,
												  z,
												  variance = spread)

		return mol_tensor

	def add_mol_gaussian(self, tensor, specie, x,y,z, variance=0.5):
		shape = tensor[specie].shape
		distances = np.zeros(shape)

		# print(shape)
		for x_i in range(shape[0]):
			for y_i in range(shape[1]):
				for z_i in range(shape[2]):
					gp_x = (x_i / (shape[0]-1) )  
					gp_y = (y_i / (shape[1]-1) )
					gp_z = (z_i / (shape[2]-1) )

					dist = ((np.abs(gp_x-x))**2+ 
							(np.abs(gp_y-y))**2+ 
							(np.abs(gp_z-z))**2)
					# print("Species: {}, x {} y {} z {} :".format(specie, gp_x, gp_y, gp_z), dist)
					distances[x_i][y_i][z_i] = dist 
		
		distances = distances / (variance**2)
		# print(distances[1])								   

		distances = np.exp(- 0.5 * distances)
		# print(distances[1])
		distances = np.power(1/(2*np.pi),3/2) * distances
		# print(distances[1])
		tensor[specie] += distances
	
		return tensor[specie]

	

	def get_voxel(self):
		return self.voxel

def Plot3D(tensor):
		fig = plt.figure(figsize = plt.figaspect(0.25))
		ax1 = fig.add_subplot(1,4,1,projection='3d')
		ax2 = fig.add_subplot(1,4,2,projection='3d')
		ax3 = fig.add_subplot(1,4,3,projection='3d')
		ax4 = fig.add_subplot(1,4,4,projection='3d')

		bl = tensor[1]
		# print(bl)
		bl[bl < bl.max()* 0.95] = 0	
		
		# print(bl)
		og = tensor[5]
		og[og < og.max() * 0.95] = 0

		ca = tensor[3]
		ca[ca < ca.max()* 0.95] = 0 

		pa = tensor[4]
		pa[pa < pa.max() *0.95] = 0


		ax1.voxels(bl, edgecolor="k", facecolor="blue")
		ax2.voxels(og, edgecolor="k", facecolor="orange")
		ax3.voxels(ca, edgecolor="k", facecolor="yellow")
		ax4.voxels(pa, edgecolor="k", facecolor="red")

		ax1.set_title("Oxygen")
		ax2.set_title("Copper")
		ax3.set_title("Carbon")
		ax4.set_title("Phosphorus")

		plt.legend()
		plt.show()

def Voxel_coords(tensor, threshold = 0.95, precision = 3 ):
	atom_species = ["H","O", "N", "C", "P",
			 "Cu","Co","Ag","Zn","Cd", "Fe"]
	string = "Atom, x , y, z \n"
	for i, specie in enumerate(atom_species):
		temp = tensor[i]

		temp[temp < temp.max() * threshold] = 0
		temp = np.transpose(temp.nonzero())
		temp = temp // precision 
		temp = temp * precision 

		locations = np.unique(temp, axis = 0)

		for loc in locations:
			string+=(specie+","+np.array2string(loc / (31), 
											  separator=",",
											  precision = 4)[1:-1])
			string += '\n'
	return string  

if __name__ == '__main__':
	cif_voxel = CIFtoVoxel('AHOKOX_clean.cif').get_voxel()
	# Plot3D(cif_voxel)
	vox_cords = Voxel_coords(cif_voxel)

	save_file = open("AHOKOX_reconstructed.csv", 'w')
	save_file.write(vox_cords)
	save_file.close()

	print(vox_cords)
