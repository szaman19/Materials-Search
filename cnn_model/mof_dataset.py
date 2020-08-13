import pickle
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

import Voxel_MOF


class MOFDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        path = Path(path)
        with path.open("rb") as f:
            self.data: List[Voxel_MOF] = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dic = {}
        dic['data'] = self.data[idx].loc_tensor
        lp = self.data[idx].lattice_params
        
        a_mean = 16.5824 
        b_mean = 18.1350 
        c_mean = 20.0291

        a_std = 8.6663 
        b_std = 7.9525
        c_std = 10.3119
        mean_tensor = torch.tensor([a_mean, b_mean, c_mean]) 
        std_tensor = torch.tensor([a_std, b_std, c_std])

        dic['lattice_params'] = [ lp['a'] , lp['b'], lp['c']]
        dic['lattice_params'] = (torch.tensor(dic['lattice_params']) - mean_tensor ) / std_tensor
        dic['metadata'] = self.data[idx].grid_metadata
        return  dic

    @staticmethod
    def get_data_loader(path: str, batch_size: int):
        return torch.utils.data.DataLoader(
            MOFDataset(path),
            batch_size=batch_size,
            shuffle=True,
        )

def main():
    data_loader = MOFDataset.get_data_loader("../3D_Grid_Data/Test_MOFS.p", 64)

    for batch, mofs in enumerate(data_loader):
        print(batch, mofs['data'].shape, mofs['lattice_params'].shape)

        break


if __name__ == '__main__':
    main()
