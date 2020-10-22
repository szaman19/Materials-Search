import pickle
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

import Voxel_MOF


class MOFDataset(Dataset):
    def __init__(self, path, no_grid=False, no_loc=False,transform=None):
        self.path = path
        self.no_grid = no_grid
        self.no_loc = no_loc
        path = Path(path)
        with path.open("rb") as f:
            self.data: List[Voxel_MOF] = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.no_grid:
            return self.data[idx].loc_tensor
        elif self.no_loc:
            return self.data[idx].grid_tensor    
        else:
            return self.data[idx].data

    @staticmethod
    def get_data_loader(path: str, batch_size: int, no_grid=False, no_loc=False):
        return torch.utils.data.DataLoader(
            MOFDataset(path, no_grid=no_grid, no_loc=no_loc),
            batch_size=batch_size,
            shuffle=True,
        )


def main():
    data_loader = MOFDataset.get_data_loader("../3D_Grid_Data/Test_MOFS.p", 25)

    batch: int
    mofs: torch.Tensor
    for batch, mofs in enumerate(data_loader):
        print(batch, mofs.shape)


if __name__ == '__main__':
    main()
