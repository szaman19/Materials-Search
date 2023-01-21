## Data Generation
---

This directory holds scripts to generate the energy grids for the MOF CoRE dataset.

### Data Download

Run the download script with: 

```python
python data_download.py
```

### Energy Grid Generation

The script requires Musen Zhao's implementation of `cif2input` and `grid_gen`. 

```python
python generate_energy_grids.py
```