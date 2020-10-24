import numpy as np 
import matplotlib.pyplot as plt 
from CIFtoTensor import CIFtoTensor



cif_file = CIFtoTensor.get_cif_file()
struc = CIFtoTensor.get_pymat_struct(cif_file)
mol_tensor = CIFtoTensor.to3DTensor(struc)

print(mol_tensor.shape)