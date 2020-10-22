import os
import pandas as pd
import glob


# def test_gen():
#     properties = pd.read_csv("MOF_info.csv")
#
#     training_files = glob.glob("data/test/*.cif")
#
#     training_files = [f.strip("data/test/\\").rstrip().strip(".cif") for f in training_files]
#     counter = 0
#
#     # print(training_files[2])
#     for filenames in properties['Name_MOF_CoRE_1272']:
#         if filenames in training_files:
#             counter += 1
#         else:
#             properties = properties[properties.Name_MOF_CoRE_1272 != filenames]  # Not sure what this line does
#
#     print(counter)
#     properties.to_csv("test/properties.csv")
#

def training_gen():
    properties = pd.read_csv("MOF_info.csv")

    files = pd.read_csv("files.csv")

    files['filename'] = files['filename'].str.rstrip(".cif")
    print(files.head())
    print(properties.head())
    counter = 0

    for filenames in properties['Name_MOF_CoRE_1272']:
        # print(filenames)
        if filenames in files['filename'].values:
            counter += 1
        else:
            properties = properties[properties.Name_MOF_CoRE_1272 != filenames]

    print(counter)
    properties.to_csv("data/properties.csv")


def main():
    # test_gen()
    training_gen()


main()
#
# training_files = glob.glob("data/test/*.cif")
#
# training_files = [f.strip("data/test/\\").rstrip().strip(".cif") for f in training_files]
#
# for file in training_files:
#     print(file)