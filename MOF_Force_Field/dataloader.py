from torch_geometric.data import DataLoader, Data
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

while True:
    try:
        file = input("Enter the name or path of the dataset to read with extension included: ")

        df = pd.read_csv(str(file), skiprows=1, low_memory=False)

    except FileNotFoundError:
        print("The file you entered does not exist or you entered the name incorrectly.")
        continue
    else:
        break

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
            
            node_features = [[0 for i in range(len(run_atom))] for j in range(len(run_atom))]
            
            source_nodes = []
            target_nodes = []
            bond_dists = []
            for i in range(len(run_atom)):
                node_features[i][i] = 1
                for k in range(len(run_atom)):
                    source_nodes.append(run_atom[i])
                    target_nodes.append(run_atom[k])
                    bond_dists.append(math.sqrt(((group.x[i] - group.x[k]) ** 2) + ((group.y[i] - group.y[k]) ** 2) + (
                                (group.z[i] - group.z[k]) ** 2)))

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor([source_nodes, bond_dists], dtype=torch.long)
            node_features = torch.LongTensor(node_features)
            
            x = node_features

            y = torch.FloatTensor(group.energy.drop_duplicates())

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # 49988 number of unique runs
        
if __name__ == '__main__':
dataset = MOFDataset('../')

dataset = dataset.shuffle()
one_tenth_length = int(len(dataset) * 0.1)
train_dataset = dataset[:one_tenth_length * 8]
val_dataset = dataset[one_tenth_length*8:one_tenth_length * 9]
test_dataset = dataset[one_tenth_length*9:]

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
