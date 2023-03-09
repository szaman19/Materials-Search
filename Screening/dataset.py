
import glob
import sys
from pathlib import Path
from pymatgen.io.cif import CifParser

import click
import numpy as np
import pandas
import torch

def matched_features(csv, feature, truth):
    features = np.zeros((len(truth), 1))
    df = pandas.read_csv(csv)
    filenames = []
    for index in df.index:
        filename = df['filename'][index]
        if filename not in truth:
            continue
        idx = truth[filename]
        features[idx, 0] = df[feature][index]
        filenames.append(filename)
    return features, filenames

class Dataset(torch.utils.data.Dataset):
    def __init__(self, grid_file, link_file, csv_file, lattice_file, feature, mapping=None):
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
            self.labels = np.load(lattice_file)
        else:
            self.labels, filenames = matched_features(csv_file, feature, truth)
            
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
