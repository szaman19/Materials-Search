## Materials Search with ML

This repo consists of the various models used in the Deep Learning based Metal Organic Framework (MOF) generation project. The sub-directories include: 

1. data - scripts for data gather from CoRE MOF Databse 
2. exploratory - exploratory statistical analysis of MOFs
3. 3D_Grid_Data - code to convert cif to 3D molecular tensors
4. cnn_model - 3D CNN model for property classification and regression
5. gcn_model - Graph Convolutional model for property classification and regression
6. 3DGen_Model - 3D generative models (GAN, VAE) for materials generation





### Downloading CoRE MOF Database

***

#### Requirements: 
1. Python3 
2. Requests 


#### Download:
1. Create Virtual environment 
2. Activate your virtual environment. (Call it venv so the git automatically ignores it)
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Run the downloader with: 
```
python data_download.py
```
5. De-compress the .tar file to retrieve the .ics files 

This material is based upon work supported by the National Science Foundation under Grant No. DMR-1940243.

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
