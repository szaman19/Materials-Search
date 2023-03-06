
import glob
import os
import click
from tqdm.auto import tqdm
from pathlib import Path
from pymatgen.io.cif import CifParser

import numpy as np

def read_grids_from(files, size=32, N=1):
    i = 0
    grids = np.zeros([len(files), N]+[size]*3, dtype=np.float32)
    for file in tqdm(files, desc="reading grids"):
        if i%(size**3):
            print("Unexpected index:", i)
        with open(file) as f:
            for line in f:
                for num in line.split()[-N:]:
                    grids.flat[i] = float(num)
                    i+=1
    return grids

def matched_lattice(cif_dir, truth):
    features = np.zeros([len(truth), 6])
    cifs = Path(cif_dir).iterdir()
    for cif_filename in tqdm(list(cifs), desc="reading cifs"):
        name = cif_filename.stem
        cif_data = CifParser(cif_filename).get_structures(primitive=False)
        assert len(cif_data) == 1
        lattice = cif_data[0].lattice
        if name not in truth:
            print(name, "not found")
            continue
        features[truth[name]] = np.array([*lattice.abc, *lattice.angles])
    return features

def lattice(to_file, cif_dir, truth):
    features = matched_lattice(cif_dir, truth)
    np.save(to_file + ".lattice", features)

def reformat(directory, to_file):
    files = glob.glob("*.grid", root_dir=directory)
    names = [x.rsplit(".", 1)[0] for x in files]
    files = [os.path.join(directory, x) for x in files]
    grids = read_grids_from(files)
    np.save(to_file, grids)
    with open(to_file+".link", "w") as f:
        f.write("\n".join(names))
    return names

@click.command()
@click.argument('directory', default='./', type=click.Path(exists=True))
@click.argument('cif-directory', default='./', type=click.Path(exists=True))
@click.argument('name', default='grids', type=click.Path())
@click.option('--skip', '-s', is_flag=True)
def main(directory, cif_directory, name, skip):
    if not skip:
        names = reformat(directory, name)
    else:
        names = Path(name+".link").read_text().split("\n")
    link = {x: i for i, x in enumerate(names)}
    lattice(name, cif_directory, link)

if __name__ == "__main__":
    main()
