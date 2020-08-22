from torch_geometric.data import DataLoader, Data
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('FIGXAU_V2.csv',skiprows=1, low_memory=False)
df.columns = ['atom','x','y','z', 'energy', 'run']


df['x'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['x'], inplace=True)
df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)
df['z'] = df['z'].astype(float)


class MOFDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MOFDataset, self).__init__(root, transform, pre_transform)
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
        grouped = df.groupby('run')
        for run, group in tqdm(grouped):
            run_atom = LabelEncoder().fit_transform(group.atom)
            group = group.reset_index(drop=True)
            group['run_atom'] = run_atom
            node_features = group.run_atom.drop_duplicates().values
            # node_features = group.loc[group.run == run, ['run_atom', 'atom']].sort_values(
            #     'run_atom').atom.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)

            source_nodes = []
            target_nodes = []
            bond_dists = []
            for i in range(len(run_atom)):
                for k in range(len(run_atom)):
                    source_nodes.append(run_atom[i])
                    target_nodes.append(run_atom[k])
                    bond_dists.append(math.sqrt(((group.x[i] - group.x[k]) ** 2) + ((group.y[i] - group.y[k]) ** 2) + (
                                (group.z[i] - group.z[k]) ** 2)))

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor([source_nodes, bond_dists], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.energy])

            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # 49988 number of unique runs


dataset = MOFDataset('../')

dataset = dataset.shuffle()
one_tenth_length = int(len(dataset) * 0.1)
train_dataset = dataset[:one_tenth_length * 8]
val_dataset = dataset[one_tenth_length*8:one_tenth_length * 9]
test_dataset = dataset[one_tenth_length*9:]
# print(len(train_dataset), len(val_dataset), len(test_dataset))

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
