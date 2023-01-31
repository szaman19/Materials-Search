
import glob
import os
import sys

import click
import numpy as np
import pandas
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, grid_file, link_file, csv, feature):
        super().__init__()
        self.grids = np.load(grid_file)
        self.labels = np.zeros(len(self.grids))
        with open(link_file) as f:
            links = f.read().split()
        truth = {}
        for idx, link in enumerate(links):
            truth[link] = idx
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
