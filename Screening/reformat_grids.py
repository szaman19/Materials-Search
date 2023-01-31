
import glob
import os
import click

import numpy as np

def read_grids_from(files, size=32, N=1):
    i = 0
    grids = np.zeros([len(files), N]+[size]*3, dtype=np.float32)
    for file in files:
        if i%(size**3):
            print("Unexpected index:", i)
        with open(file) as f:
            for line in f:
                for num in line.split()[-N:]:
                    grids.flat[i] = float(num)
                    i+=1
    return grids

def reformat(directory, to_file):
    files = glob.glob("*.grid", root_dir=directory)
    names = [x.rsplit(".", 1)[0] for x in files]
    files = [os.path.join(directory, x) for x in files]
    grids = read_grids_from(files)
    np.save(to_file, grids)
    with open(to_file+".link", "w") as f:
        f.write("\n".join(names))

@click.command()
@click.argument('directory', default='./', type=click.Path(exists=True))
@click.argument('name', default='grids', type=click.Path())
def main(directory, name):
    reformat(directory, name)

if __name__ == "__main__":
    main()
