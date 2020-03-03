from CIFtoTensor import CIFtoTensor
import glob

def main():
    files = glob.glob("data/training/*.cif")
    
    for f in files:
        cif_file = CIFtoTensor.get_cif_file(f)
        struc = CIFtoTensor.get_pymat_struct(cif_file)
        print(struc)
        mol_tensor = CIFtoTensor.to3DTensor(struc)
        print(mol)
        print(f)
        break

main()
