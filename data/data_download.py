import requests 
import tarfile



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
	url_cif = 'https://zenodo.org/record/3677685/files/2019-11-01-ASR-public_12020.tar.gz?download=1'
	cif_tar_name = '2019-11-01-ASR-public_12020.tar.gz'
	downloader(url_cif, cif_tar_name)

	tar = tarfile.open(cif_tar_name	, "r:gz")
	tar.extractall()
	tar.close()

	print('Beginning csv download')
	url_csv = "https://zenodo.org/record/3677685/files/2019-11-01-ASR-public_12020.csv?download=1"
	csv_name = "2019-11-01-ASR-public_12020.csv"
	downloader(url_csv, csv_name)
	
	print('Beginning csv download')
	url_csv = "https://zenodo.org/record/3677685/files/2019-11-01-ASR-internal_14142.csv?download=1"
	csv_name = "2019-11-01-ASR-internal_14142.csv"
	downloader(url_csv, csv_name)
	

if __name__ == "__main__":        
	main()
