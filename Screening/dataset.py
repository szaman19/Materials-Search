
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

def load_features(feature_names, lattice_file, csv_file):
    feature_map = defaultdict(list)
    lattice_map = np.load(lattice_file, allow_pickle=True).item() if "lattice" in feature_names else None
    df = pandas.read_csv(csv_file)
    for feature_name in feature_names:
        if feature_name == "lattice":
            for filename, lattice in lattice_map.items():
                feature_map[filename].extend(lattice)
        else:
            for filename, feature in df_features(df, feature_name):
                feature_map[filename].append(feature)
    return feature_map

class Dataset(torch.utils.data.Dataset):
    def __init__(self, grid_file, csv_file, lattice_file, feature_names, transform=lambda x: x):
        super().__init__()
        grid_map = np.load(grid_file, allow_pickle=True).item()
        grid_names = grid_map.keys()
        grids = [np.float32(transform(grid_map[x])) for x in grid_names]
        self.grids = np.array(grids)
        
        feature_names = feature_names.split()
        feature_map = load_features(feature_names, lattice_file, csv_file)
        print("available features:", len(feature_map), "available grids:", len(grid_map))
        missing_features = 0
        feature_size = max(*(len(x) for x in feature_map.values()))
        feature_values = []
        for filename in grid_names:
            features = feature_map.get(filename, [])
            if len(features) == feature_size:
                feature_values.append(np.array(features))
            else:
                print(f"Only {len(features)}/{feature_size} features for {filename}")
                missing_features += 1
                feature_values.append(np.zeros(feature_size))
        self.labels = np.array(feature_values)
        print(missing_features, "missing features")            
            
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
