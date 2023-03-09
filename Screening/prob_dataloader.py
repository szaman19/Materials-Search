import torch
from pymatgen.io.cif import CifParser
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import click

from grid_generator import calculate_supercell_coords, GridGenerator
from torch import Tensor

# data2 = torch.load("./fixed_training_grids.pt") # re-ranged energy grids
directory2 = '/home/szaman5/Materials-Search/data/energy_grids/'
directory = '/home/szaman5/Materials-Search/data/structure_10143/'


def read_grids_from(directory, size=32, N=1):
    i = 0
    files = os.listdir(directory)
    grids = np.zeros([len(files), N]+[size]*3, dtype=np.float32)
    for file in files:
        if i%(size**3):
            print("Unexpected index:", i)
        with open(directory+file) as f:
            for line in f:
                for num in line.split()[-N:]:
                    grids.flat[i] = float(num)
                    i+=1
    return grids


def load_probability(unit_cell_coords, lattice, position_supercell_threshold: float, position_variance: float) -> Tensor:
    transformation_matrix = lattice.matrix.copy()
    a, b, c = lattice.abc
    super_cell_coords = calculate_supercell_coords(unit_cell_coords, threshold=position_supercell_threshold)
    weights = np.ones((len(super_cell_coords), 1))
    super_cell_coords = np.hstack((weights, super_cell_coords))
    torch_coords = torch.from_numpy(super_cell_coords).float()
    return GridGenerator(32, position_variance).calculate(torch_coords, a, b, c, transformation_matrix)

def cif_probability(file):
    parser = CifParser(file)
    structures = parser.get_structures(primitive=False)
    assert len(structures) == 1
    structure = structures[0]
    lattice = structure.lattice

    structure2 = structure.copy()
    structure2.remove_species(['H', 'C', 'N', 'P', 'O'])
    structure.remove_species(structure2.species)
    coords1 = structure.frac_coords
    coords2 = structure2.frac_coords
    if len(coords1) == 0 or len(coords2) == 0:
        print(filename)
        return
    organics = load_probability(coords1, lattice, 0.4, 0.2)
    metals = load_probability(coords2, lattice, 0.4, 0.2)
    return organics, metals

def data_parser():
    energy_grids = np.load("data/grids.npy", allow_pickle=True).item()
    grouped_grids = {}
    for filename in tqdm(energy_grids):
        data = np.zeros([3, 32, 32, 32])
        f = Path(directory, filename+'.cif')
        if not f.is_file():
            print("Could not find", filename)
            continue
        organics, metals = cif_probability(f)
        data[0] = energy_grids[filename]
        data[1] = organics
        data[2] = metals
        grouped_grids[filename] = data

    np.save("data/probability", grouped_grids)

    print("Completed")

if __name__ == "__main__":
    data_parser()

