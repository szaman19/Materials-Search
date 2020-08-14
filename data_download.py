import requests 


def downloader(link, file_name):
	url = link
	r = requests.get(url)

	with open(file_name, 'wb') as f:
		f.write(r.content)
		f.close()
	if (r.status_code == "200"):
		print("Completed download")

def main():
	print('Beginning cif download')
	url_cif = 'https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.tar?download=1'
	cif_tar_name = '2019-07-01-ASR-public_12020.tar'
	downloader(url_cif, cif_tar_name)
	
	print('Beginning csv download')
	url_csv = "https://zenodo.org/record/3370144/files/2019-07-01-ASR-public_12020.csv?download=1"
	csv_name = "2019-07-01-ASR-public_12020.csv"
	downloader(url_csv, csv_name)
	
        
main()
