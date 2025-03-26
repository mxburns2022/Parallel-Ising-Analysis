# Data and Scripts for "Limitations in Parallel Ising Machine Networks: Theory and Practice"

## Data and Figures
All data used to generate plots can be found in `data`. Note that some CSVs contain only summarized statistics to save space. For instance, the `*_ttt.csv` files only contain the computed "time-to-target" values.

Plotting code can be found in the `notebooks` folder, where the notebooks are named according to the figure that they generate. Note that we assume the user has STIX fonts installed, otherwise the figures will default to `matplotlib`'s default font. We also include data/plots that did not make it to the final version of the paper which compare linear and Kuramoto model behavior. 


## Experiment Code
Scripts to run small-scale experiments can be found in `par_im`. `wasserstein.jl` was used to generate data for Figures 5 and 6 (numerical experiments illustrating derived $W_1$ bounds). Data from `multi_replica.py` and `single_replica.py` were not used in the final version of the paper, however they provide interesting insight into the behavior of the linear model and nonlinear Kuramoto model, hence we include the data/plots for those interested (and potentially for future use).

A Julia script to run the small-scale Wasserstein experiments in contained in `par_im/*`. 

2000-node GSet experiments were run using a separate C++ code on a HPC cluster, however the data is included in `data/*`.

<!-- Yes, we used separate Julia, Python, and C++ simulators to run experiments. I greatly enjoy programming and learning new languages. -->