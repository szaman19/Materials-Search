import pickle
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

import numpy as np

from mof_dataset import MOFDataset
from util.rotations import Rotations


def normalize(k):
    return np.sign(k) * np.log(abs(k) + 1)


class MOFDatasetV2(Dataset):
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data: Tensor = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_data_loader(path: str, batch_size: int, shuffle=True):
        return torch.utils.data.DataLoader(
            MOFDatasetV2(path),
            batch_size=batch_size,
            shuffle=shuffle,
        )


def main():
    file_type = "Training" if False else "Test"

    data_loader = MOFDataset.get_data_loader(f"_data/{file_type}_MOFS.p", batch_size=1)

    result_list = []

    batch: int
    mofs: torch.Tensor
    for batch, mofs in enumerate(data_loader):
        result_list.append(mofs)
        for rotation in Rotations.rotate_3d(mofs[0][0]):
            result_list.append(rotation.unsqueeze(0).unsqueeze(0))

    result = torch.cat(result_list)

    output_path = Path(f"_data/{file_type}_MOFS_v2.p")
    with output_path.open("wb+") as f:
        pickle.dump(result, f, protocol=4)

    print(result.shape)
    print("DONE!")


if __name__ == '__main__':
    main()
