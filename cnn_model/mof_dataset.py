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
        return [self.data[idx].data,self.data[idx].lattice_params, self.data[idx].grid_metadata]

    @staticmethod
    def get_data_loader(path: str, batch_size: int):
        return torch.utils.data.DataLoader(
            MOFDataset(path),
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
