import os
import random
import tarfile
from os import path
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric
from pymatgen.io.cif import CifParser
from torch_geometric.data import InMemoryDataset


class BinaryDataSet(InMemoryDataset):

    def __init__(self, train=True):
        root = os.getcwd()
        self.train = train
        self.properties_file = 'properties.csv'
        self.data_file = 'data.tar'
        self.processed_test_data = 'test_data.pt'
        self.processed_training_data = 'training_data.pt'
        self.data_file_path = path.join(root, 'raw', self.data_file)
        self.properties_file_path = path.join(root, 'raw', self.properties_file)
        super(BinaryDataSet, self).__init__(root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_training_data if train else self.processed_test_data)

    @property
    def raw_file_names(self):
        return [self.properties_file, self.data_file]

    @property
    def processed_file_names(self):
        return [self.processed_test_data, self.processed_training_data]

    @staticmethod
    def validate_caller():
        if __name__ != '__main__':
            print("Run this script separately to generate necessary files")
            exit(1)

    def download(self):
        self.validate_caller()
        print('Downloading properties CSV...')
        self.download_file(url='https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.csv?download=1',
                           file_name=self.properties_file_path)
        print('Downloading CIF data...')
        self.download_file(url='https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.tar?download=1',
                           file_name=self.data_file_path)

        print("Extracting tar...")
        with tarfile.open(self.data_file_path) as tar:
            tar.extractall(self.raw_dir)

        print("Generating training set and test set...")

        source_dir = path.join(self.raw_dir, "structure_11660")
        train_dir = path.join(self.raw_dir, "training")
        test_dir = path.join(self.raw_dir, "test")

        os.mkdir(train_dir)
        os.mkdir(test_dir)

        for cif_file_path in Path(source_dir).iterdir():
            cif_file = cif_file_path.name
            num = random.random()
            source_file = path.join(source_dir, cif_file)
            if num < .79:
                os.rename(source_file, path.join(train_dir, cif_file))
            else:
                os.rename(source_file, path.join(test_dir, cif_file))

        print("Done!")

    def process(self):
        self.validate_caller()
        data_list = []
        print("Creating binary PyTorch dataset...")

        labels = pd.read_csv(self.properties_file_path)
        for data_type in ['training', 'test']:
            output_file = self.processed_training_data if data_type == 'training' else self.processed_test_data
            directory = Path(path.join(self.raw_dir, data_type))

            counter = 0
            total_files = 0
            for _ in directory.iterdir():
                total_files += 1

            for file in directory.iterdir():
                structure = self.cif_structure(str(file))
                distance_matrix = structure.distance_matrix

                graph = nx.from_numpy_matrix(distance_matrix.astype(np.double))
                num_nodes = distance_matrix.shape[0]

                data = torch_geometric.utils.from_networkx(graph)
                data.x = torch.ones(num_nodes, 1)
                data.y = labels['LCD'][counter]
                data_list.append(data)

                print("Elements loaded: ", counter, "/", total_files)
                counter += 1

            data, slices = self.collate(data_list)
            torch.save((data, slices), output_file)

    @staticmethod
    def download_file(url, file_name):
        r = requests.get(url)

        with open(file_name, 'wb') as f:
            f.write(r.content)
        if r.status_code == "200":
            print("Completed download")

    @staticmethod
    def cif_structure(file_name):
        parser = CifParser(file_name)
        structure = parser.get_structures()[0]
        return structure


if __name__ == '__main__':
    BinaryDataSet()
