import pickle
import MOFDataset


def main():

	tdl = MOFDataset.MOFDataset(train=True).get_data()

	mof_gen = open("MOF_GENERATOR_32.p", 'wb')

	pickle.dump(tdl, mof_gen)

main()