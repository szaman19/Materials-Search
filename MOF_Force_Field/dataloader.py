from torch_geometric.data import DataLoader, Data
import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd



class MOFDataset(InMemoryDataset):
    def __init__(self,
                 file_name,  
                 root, 
                 transform=None, 
                 pre_transform=None):
        self.df = pd.read_csv(file_name)
        self.df['x'].replace(' ', np.nan, inplace=True)
        self.df.dropna(subset=['x'], inplace=True)
        self.df['x'] = df['x'].astype(float)
        self.df['y'] = df['y'].astype(float)
        self.df['z'] = df['z'].astype(float)


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
        grouped = self.df.groupby('run')
        for run, group in tqdm(grouped):
            run_atom = LabelEncoder().fit_transform(group.atom)
            group = group.reset_index(drop=True)
            group['run_atom'] = run_atom
            node_features = group.run_atom.values
            
            node_features = node_features.reshape(len(node_features), 1)
            node_features = OneHotEncoder(sparse=False).fit_transform(node_features)
            node_features = torch.LongTensor(node_features)
            
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
