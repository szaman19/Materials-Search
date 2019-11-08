from scipy import stats

def main():
	values = open("vals.log",'r')
	actual = []
	predicted = []

	for each in values:
		each = each.rstrip()
		temp = each.split(",")
		predicted.append(float(temp[0]))
		actual.append(float(temp[1]))

	print(stats.pearsonr(actual, predicted))


main()