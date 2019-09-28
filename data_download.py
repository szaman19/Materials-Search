import requests 

def main():
	print('Beginning file download with requests')

	url = 'https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.tar?download=1'
	r = requests.get(url)

	with open('data/2019-07-01-ASR-public_12020.tar', 'wb') as f:
		f.write(r.content)
	# Retrieve HTTP meta-data
	print(r.status_code)
	print(r.headers['content-type'])
	print(r.encoding)
main()