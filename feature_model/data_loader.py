from pymatgen.io.cif import CifParser
from time import time
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from sklearn.utils import check_random_state
import random

labels = pd.read_csv("2019-07-01-ASR-public_12020.csv")

directory_list = ["mofdatabase_big/" + file + ".cif" for file in labels['filename'].values]

with open("mof4_dataset/mof4_info.txt", "w") as file1:
    for i in range(len(directory_list)):
        file1.write(str(labels['LCD'][i])+"\n")

for i, filename in enumerate(directory_list):
    # i += 5679
    b = str(i)
    t7 = time()
    parser = CifParser(filename)

    structure = parser.get_structures()[0]  # Might give warnings precision dwai

    frac_coordinates = structure.frac_coords  # N x 3 array. N is number of atoms

    new1 = structure.distance_matrix

    frac_trans = frac_coordinates.T

    mds = manifold.MDS(2, max_iter=100, n_init=1, n_jobs=-1, dissimilarity='precomputed')
    trans_data = mds.fit_transform(new1).T

    fig = plt.figure(figsize=(9, 9))
    plt.scatter(trans_data[0], trans_data[1], c="000000", cmap="gray", s=5)

    plt.box(on=None)
    # plt.xticks(x, "")
    # plt.yticks(y, "")
    plt.axis('off')

    t6 = time()
    print("run time: %.2g sec" % (t6 - t7))

    plt.savefig('mof4_dataset/' + str(b + '_mof4.png').zfill(14), bbox_inches='tight')
    plt.clf()


