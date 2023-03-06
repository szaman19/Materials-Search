
import glob
import sys
from pathlib import Path
from pymatgen.io.cif import CifParser

import click
import numpy as np
import pandas
import torch

def matched_features(csv, truth):
    features = np.zeros(len(truth))
    df = pandas.read_csv(csv)
    missed = 0
    for index in df.index:
        filename = df['filename'][index]
        if filename not in truth:
            missed += 1
            # print(filename, "not found")
            continue
        idx = truth[filename]
        self.labels[idx] = df[feature][index]
    print("missed", missed)
    return features

class Dataset(torch.utils.data.Dataset):
    def __init__(self, grid_file, link_file, csv_lattice, feature, mapping=None):
        super().__init__()
        grids = list(np.float32(np.load(grid_file)))
        if mapping:
            grids = [mapping(x) for x in grids]
        self.grids = np.array(grids)
        with open(link_file) as f:
            links = f.read().split()
        truth = {}
        for idx, link in enumerate(links):
            truth[link] = idx
        if feature == "lattice":
            self.labels = np.load(csv_lattice)
        else:
            self.labels = matched_features(csv_cif, truth)
            
    def __len__(self):
        return len(self.grids)
    def __getitem__(self, idx):
        return self.grids[idx], self.labels[idx]

@click.command()
@click.argument('grid', type=click.Path(exists=True))
@click.argument('link', type=click.Path(exists=True))
@click.argument('csv', type=click.Path(exists=True))
@click.argument('feature', default="LCD")
def main(grid, link, csv, feature):
    data = Dataset(grid, link, csv, feature)
    grid, label = data[0]
    print(len(data))
    print(grid.shape)
    print(label)

if __name__ == '__main__':
    main()
    # print(data[0][0])
