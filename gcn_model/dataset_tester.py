import pickle
import MOFDataset


def main():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	tdl = MOFDataset.MOFDataset(train=True).get_data()

	mof_gen = open("MOF_GENERATOR_32.p", 'wb')

	pickle.dump(tdl, mof_gen)

main()