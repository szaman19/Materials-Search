# MOF data
This folder is expiriments on high throughput screening of MOFs

### Preparing Data

Data is prepared so it can be loaded quickly instead of parsing a bunch of files beforehand.
Prepare data using `python3 reformat_grids.py data/energy_grids data/structures`

### Training to predict lattice parameters

Training to predict lattice parameters is done using `python3 lattice_train.py`.

### Using the model

lattice_net.ipynb is experiments with the trained model