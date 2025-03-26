# Data and Scripts for "Limitations in Parallel Ising Machine Networks: Theory and Practice"

## Data and Figures
All data used to generate plots can be found in `data`. Note that some CSVs contain only summarized statistics to save space. For instance, the `*_ttt.csv` files only contain the computed "time-to-target" values.

Plotting code can be found in the `notebooks` folder, where the notebooks are named according to the figure that they generate. Note that we assume the user has STIX fonts installed, otherwise the figures will default to `matplotlib`'s default font. 


## Experiment Code
Scripts to run small-scale experiments can be found in `par_im`. `wasserstein.jl` was used to generate data for Figures 5 and 6 (numerical experiments illustrating derived $W1$ bounds). 
A Julia script to run the small-scale Wasserstein experiments in contained in `par_im/*`. 

GSet experiments were run using a separate C++ code on a HPC cluster, however the data is included in `data/*`. The scripts used to produce plots are located in `notebooks/*`. Each notebook is labeled according to the figure that it generates  
