
import glob
import os
import click
from tqdm.auto import tqdm
from pathlib import Path
from pymatgen.io.cif import CifParser

import numpy as np

def read_grids(files, size=32):
    grids = {}
    for file in tqdm(files, desc="reading grids"):
        with open(file) as f:
            grid = np.zeros([1]+[size]*3, dtype=np.float32)
            for i, line in enumerate(f):
                grid.flat[i] = float(line.split()[-1])
            grids[Path(file).stem] = grid
    return grids

def read_lattices(cif_dir):
    features = {}
    cifs = Path(cif_dir).iterdir()
    for cif_filename in tqdm(list(cifs), desc="reading cifs"):
        name = cif_filename.stem
        cif_data = CifParser(cif_filename).get_structures(primitive=False)
        assert len(cif_data) == 1
        lattice = cif_data[0].lattice
        features[name] = np.array([*lattice.abc, *lattice.angles])
    return features

def lattice(cif_directory, to_file):
    lattices = read_lattices(cif_directory)
    np.save(to_file + ".lattice", lattices)
    return lattices

def reformat(directory, to_file):
    """Reformat ascii-based grids in directory to a .npy file and .link file
    
    The .link file contains the names of all cifs in the order they appear in the .npy file
    """
    files = glob.glob("*.grid", root_dir=directory)
    files = [Path(directory, x) for x in files]
    grids = read_grids(files)
    np.save(to_file, grids)
    return grids

@click.command()
@click.argument('directory', default='./', type=click.Path(exists=True))
@click.argument('cif-directory', default='./', type=click.Path(exists=True))
@click.argument('name', default='grids', type=click.Path())
@click.option('--skip-energy', is_flag=True)
@click.option('--skip-latice', is_flag=True)
def main(directory, cif_directory, name, skip_energy, skip_lattice):
    if not skip_energy:
        reformat(directory, name)
    if not skip_lattice:
        lattice(cif_directory, name)

if __name__ == "__main__":
    main()
