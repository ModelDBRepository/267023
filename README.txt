This directory contains scripts for simulating the grid-cell-to-place-cell transformation model
with AD-related synapse loss related to the article: 

"A computational grid-to-place-cell transformation model indicates a synaptic driver of place cell
impairment in early-stage Alzheimer's Disease". Natalie Ness, Simon R. Schultz; 2020, bioRvix. 

The simulation was written with Python 3.7.

To perform simulations, run the script 'model_3.py'. The script runs a 'wildtype' (no synaptic loss) 
simulation over 30 days. The script outputs the place cell array and generates summary graphs showing 
the total number of place cell, number of new place cells, mean place field width and recurrence probability
of place cells and active cells. 

The cell entitled 'Run simulation' contains various useful parameters, which may be adjusted. 
To change the time period over which the simulation is run, adjust "samples" in line 914, where the 
first value in np.arange() gives the starting day (must be 0), the second value gives the 
maximum number of days, and the last value gives the interval of days at which place cell properties
are recorded. In the case of long simulations or high sampling rate, we recommend exporting the place cell
array at each iteration, rather than updating the place cell dict to avoid memory issues.

To implement excitatory synaptic loss, set gc_syn_loss=True.
To implement inhibitory synaptic loss, set inh_syn_loss=True.



