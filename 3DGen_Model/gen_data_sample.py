from CIFtoTensor import CIFtoTensor
import glob
import pickle

def main():
    files = glob.glob("data/training/*.cif")
    
    counter = 0
    file_num = 0

    tensors = []
    for f in files:
        cif_file = CIFtoTensor.get_cif_file(f)
        struc = CIFtoTensor.get_pymat_struct(cif_file)
        
        mol_tensor = CIFtoTensor.to3DTensor(struc, normalize=134)
        
        print(file_num * 320 + counter, "/", len(files))
        tensors.append(mol_tensor)
        counter +=1 
        if (counter == 320):
        	pickle.dump(tensors, "training_mol_tensors_"+str(file_num)+".p")
        	file_num +=1
        	counter = 0 
        	tensors = []
        else if (file_num*320 + counter == len(files)):
        	pickle.dump(tensors, "training_mol_tensors_"+str(file_num)+".p")

main()
