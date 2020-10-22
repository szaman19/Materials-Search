from torch_geometric.data import DataLoader, Data
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import math
import numpy as np
from scipy.spatial import distance

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import torch_geometric.utils as data_utils

def get_torch_data(df, threshold=3):
    df = df.dropna(axis=1, how='all')
    df.columns = ['name', 'pld', 'lcd', 'ucv', 'vf', 'henry_const', 'D0inA', 'D0inB', 'D0inC']

    mols = df['name'].values

    hen_const = np.array([-1 * df['henry_const'].values])
    mols = np.expand_dims(mols, axis=1)

    one_hot_encoding = OneHotEncoder(sparse=False).fit_transform(mols)
    attributes = df[['x(angstrom)', 'y(angstrom)', 'z(angstrom)']].values

    edge_index = None
    edge_attr = None

    while True:
        dist = distance.cdist(coords, coords)
        dist[dist > threshold] = 0
        dist = torch.from_numpy(dist)
        edge_index, edge_attr = data_utils.dense_to_sparse(dist)
        edge_attr = edge_attr.unsqueeze(dim=1).type(torch.FloatTensor)
        edge_index = torch.LongTensor(edge_index)
        if (data_utils.contains_isolated_nodes(edge_index, num_nodes=13)):
            threshold += 0.5
        else:
            break

    x = torch.from_numpy(one_hot_encoding).type(torch.FloatTensor)
    y = torch.from_numpy(energy).type(torch.FloatTensor)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


class MOFDataset(InMemoryDataset):
    def __init__(self,
                 file_name,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.df = pd.read_csv(file_name)

        super(MOFDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.pre_filter = pre_filter

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['bonds.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        # process by run
        grouped = self.df.groupby('run')
        for run, group in tqdm(grouped):
            group = group.reset_index(drop=True)
            data_list.append(get_torch_data.remote(group[1:]))

        if (self.pre_filter):
            data_list = [x for x in data_list if self.pre_filter(x)]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # 49988 number of unique runs


if __name__ == '__main__':
    dataset = MOFDataset('MOF_info.csv', '.')

    dataset = dataset.shuffle()
    one_tenth_length = int(len(dataset) * 0.1)
    train_dataset = dataset[:one_tenth_length * 8]
    val_dataset = dataset[one_tenth_length * 8:one_tenth_length * 9]
    test_dataset = dataset[one_tenth_length * 9:]

    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)