
import glob
import sys
from pathlib import Path
from pymatgen.io.cif import CifParser
from collections import defaultdict

import click
import numpy as np
import pandas
import torch

def df_features(df, feature):
    if feature not in df:
        print(feature, "not in data, available features:", ", ".join(df.columns))
        return
    for index in df.index:
        filename = df['filename'][index]
        val = df[feature][index]
        yield filename, val

def load_features(feature_names, lattice_file, csv_file, file_indexes):
    feature_map = defaultdict(list)
    lattice = np.load(lattice_file) if "lattice" in feature_names else None
    df = pandas.read_csv(csv_file)
    for feature_name in feature_names:
        if feature_name == "lattice":
            for i, file in enumerate(file_indexes):
                assert i == file_indexes[file]
                feature_map[file].extend(lattice[i])
        else:
            for filename, feature in df_features(df, feature_name):
                feature_map[filename].append(feature)
    return feature_map

class Dataset(torch.utils.data.Dataset):
    def __init__(self, grid_file, link_file, csv_file, lattice_file, feature_names, mapping=None):
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
        feature_names = feature_names.split()
        feature_values = [None]*len(grids)
        feature_map = load_features(feature_names, lattice_file, csv_file, truth)
        print("available features:", len(feature_map), "available grids:", len(truth))
        missing_grids = 0
        feature_size = 0
        for filename, features in feature_map.items():
            # assert(len(features) == len(feature_names))
            if filename not in truth:
                missing_grids += 1
                continue
            feature_values[truth[filename]] = np.array(features)
            if feature_size:
                assert(len(features) == feature_size)
            else:
                feature_size = len(features)
        self.labels = np.array(feature_values)
        print(missing_grids, "missing grids")            
            
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
