
def main():
	f = open("example.inp", 'r')
	file_name = f.readline()
	Creation_time = f.readline()
	Version = f.readline()

	f.readline()
	NMAX = f.readline()
	f.readline()
	VECTORS = f.readline()
	f.readline()
	ANGLES = f.readline()
	f.readline()
	PARAMS = f.readline()

	f.readline()

	NUM_ATOMS = int(f.readline().strip("\n"))

	COL_VALS = f.readline()

	for x in range(NUM_ATOMS):
		print(f.readline())
		



	#print(f.readline())
	#print(f.readline())
	# print(f[5])


if __name__ == '__main__':
	main()