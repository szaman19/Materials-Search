from glob import glob
import os.path as osp
from tqdm import tqdm 
import subprocess as sp


def main():

  cur_dir = osp.dirname(osp.realpath(__file__))
  struct_dir = osp.join(cur_dir, "structure_10143")
  inp_dir = osp.join(cur_dir, "inp_grids")
  energy_dir = osp.join(cur_dir, "energy_grids")

  UFF_loc = osp.join(cur_dir, "MOFGAN/data_ff_UFF")

  cif_files = glob(struct_dir+"/*.cif")
  cif_names = [x.split("/")[-1][:-4] for x in cif_files]
  
  num_files = len(cif_files)

  num_concurrent_processes = 38
  for _blocks in tqdm(range(0, num_files, num_concurrent_processes)):
      counter = 0
      procs = []
      while (counter < num_concurrent_processes and counter + _blocks < num_files):
        
        # generate inp file
        cif_name = cif_names[_blocks + counter]
        cif_file = cif_files[_blocks + counter]
        inp_file = osp.join(inp_dir, cif_name+".inp")
        grid_file = osp.join(energy_dir, cif_name+".grid")
        # print(inp_file, cif_file, UFF_loc)
        p = sp.Popen(["./cif2input", cif_file, UFF_loc, inp_file])    
        p.wait()

        procs.append(sp.Popen(["./grid_gen", inp_file, grid_file]))

        counter += 1
        
      exit_codes = [p.wait() for p in procs]
      print(exit_codes)

  print(num_files)



if __name__ == "__main__":
  main()