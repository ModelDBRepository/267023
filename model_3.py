#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Natalie Ness, 2021

Grid cell-to-place cell transformation with AD-related synapse loss simulation

Relating to Ness N, Schultz SR. 'A computational grid-to-place-cell transformation 
model indicates a synaptic driver of place cell impairment in early-stage Alzheimer’s Disease' 

"""
# %%

import numpy as np 
import csv
import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

# %% Define Functions 

def initialize_place_cells(n_pc, n_grids):
    """ Initialize place cell population

    Parameters
    ----------
    n_pc : int
        Number of place cells.
    n_grids : int
        Number of grid cells.

    Returns
    -------
    place_cells : dict
        Dictionary with initialised place cell synapses and synaptic weights arrays.

    """
    #dictionary to store place_cell info 
    place_cells = {}
    #Number GC-to-PC synapses, with 1200 synapses in a network of 311,500 place cells. Scaled depending on number of PCs.
    n_syn = int(np.ceil(1200*(n_pc/311500)))
    #Randomly assign synapses between GCs and PCs 
    gc_input_binary = np.random.permutation(np.hstack((np.ones([1,n_syn]), np.zeros([1,n_grids-n_syn])))[0])
    PC_synapses = np.zeros([n_grids, n_pc])
    for i in range(0,n_pc):
        PC_synapses[:,i] = np.random.permutation(gc_input_binary[:])
    place_cells['synapses'] = PC_synapses
    #initialise synaptic weights randomly from synaptic strength pool 
    weights = np.random.choice(synaptic_strength_pool, size=(n_grids, n_pc), replace=True)
    weights = np.multiply(PC_synapses, weights) 
    place_cells['weights'] = weights
    return place_cells

def turnover(place_cells):
    """ Synaptic turnover of grid cell-to-place cell synapses

    Parameters
    ----------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.

    Returns
    -------
    place_cells : dict
        Dictionary with place cell  with updated synapses and synaptic weights arrays.

    """
    synapses = place_cells['synapses']
    weights = place_cells['weights']
    n_pc = synapses.shape[1]
    n_turnover = int(np.ceil(114*(n_pc/311500)))  #number of synapses to be replaced based on 114 in a network of 311,500 PCs

    for c in range(synapses.shape[1]):
        #turnover synapses
        lost_synapses = np.where(synapses[:,c] == 1.0)[0] 
        #check if there are enough synapses to turnover n_turnover synapses despite AD-related loss
        if len(lost_synapses) >= n_turnover: 
            lost_synapses = np.random.choice(lost_synapses, size=n_turnover, replace=False)
            synapses[lost_synapses,c] = 0.
            weights[lost_synapses,c] = 0. 
            new_synapses = np.where(synapses[:,c] == 0.0)[0]
            new_synapses = np.random.choice(new_synapses, size=n_turnover, replace=False)
            synapses[new_synapses,c] = 1.
            #initialise new weights for naive synapses 
            weights[new_synapses, c] = np.random.choice(synaptic_strength_pool, size=len(new_synapses), replace=True)
        
        elif (len(lost_synapses) < n_turnover) and (len(lost_synapses) >=1):
            #set remaining synapses to 0 if less than n_turnover available 
            synapses[lost_synapses,c] = 0.
            weights[lost_synapses,c] = 0.  
            new_synapses = np.where(synapses[:,c] == 0.0)[0]
            new_synapses = np.random.choice(new_synapses, size=len(lost_synapses), replace=False)
            synapses[new_synapses,c] = 1.
            #initialise new weights for naive synapses 
            weights[new_synapses, c] = np.random.choice(synaptic_strength_pool, size=len(new_synapses), replace=True)
    place_cells['synapses'] = synapses
    place_cells['weights'] = weights
    return place_cells 


def synapse_loss(place_cells, day_count_gc=0): 
    """ AD-related excitatory synapse loss (grid cell-to-place cell synapses)

    Parameters
    ----------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.
    day_count_gc : int, optional
        Day count for excitatory synapse loss. The default is 0.

    Returns
    -------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays, reflecting loss of excitatory synapses.
    day_count_gc : int
        Day count for excitatory synapse loss.

    """
    
    synapses = place_cells['synapses']
    weights = place_cells['weights']
    n_pc = synapses.shape[1]
    n_syn_left = np.sum(synapses) #number of synapses left
    n_syn = n_pc*int(np.ceil(1200*(n_pc/311500))) #total number of synapses on day 0
    
    #determine number of synapses to be lost depending on time passed since simulation start
    timepoint = day_count_gc/30
    pr_timepoint = (day_count_gc-1)/30
    percent_synapses_uptodate = ((0.2716*(timepoint**2))-(9.0677*timepoint))/100
    percent_synapses_uptopreviousdate = ((0.2716*(pr_timepoint**2))-(9.0677*pr_timepoint))/100
    percent_per_day = percent_synapses_uptopreviousdate - percent_synapses_uptodate 
    #add time 
    day_count_gc += 1
    #number of synapses lost on this iteration
    n_loss = int(np.round(percent_per_day * n_syn))
    #if not enough synapses left, set n_loss equal to the total number of synapses left
    #this should not occur during a 365 day simulation
    if n_syn_left < n_loss:
        n_loss = n_syn_left
    
    cells_affected = np.random.choice(np.arange(n_pc), size=n_loss, replace=True)
    #eliminate synapses 
    for c in range(synapses.shape[1]):
        if c in cells_affected:
            n_SYN_affected = np.count_nonzero(cells_affected == c)
            deleted_synapses = np.where(synapses[:,c] == 1.0)[0]
            if len(deleted_synapses) > n_SYN_affected:
                deleted_synapses = np.random.choice(deleted_synapses, size=n_SYN_affected, replace=False)
            synapses[deleted_synapses,c] = 0.
            weights[deleted_synapses,c]=0.
    place_cells['synapses'] = synapses
    place_cells['weights'] = weights 
    return [place_cells, day_count_gc]

def interneuron_synapse_turnover(in_connectivity, out_connectivity):
    """ Synaptic turnover of place cell-to-interneuron and interneuron-to-place cell synapses

    Parameters
    ----------
    in_connectivity : list
        List of place cell-to-interneuron synapses. 
    out_connectivity : list
        List of interneuron-to-place cell synapses. 

    Returns
    -------
    in_connectivity : list
        List of place cell-to-interneuron synapses reflecting synaptic turnover. 
    out_connectivity : list
        List of interneuron-to-place cell synapses reflecting synaptic turnover. 

    """
    n_interneurons = len(in_connectivity)
    #get number of synapses for turnover per interneuron
    n_in = len(in_connectivity[0])
    n_out = len(out_connectivity[0])
    n_turnover_in = int(np.round(n_in-(n_in*np.exp(-1/10)))) #determined based on decay model N0-Nt
    n_turnover_out = int(np.round(n_out-(n_out*np.exp(-1/10))))
    
    for i in range(n_interneurons): 
        cell_pool = np.arange(n_pc)
        in_pool = [x for x in cell_pool if x not in in_connectivity[i]]
        out_pool = [x for x in cell_pool if x not in out_connectivity[i]]
        
        idx_lost_in = np.random.choice(np.arange(len(in_connectivity[i])), size=n_turnover_in, replace=False)
        idx_new_in = np.random.choice(in_pool, size=n_turnover_in, replace=False)
        
        #check if there is enough synapses left for determined level of turnover 
        if n_turnover_out <= len(out_connectivity[i]):
            idx_lost_out = np.random.choice(np.arange(len(out_connectivity[i])), size=n_turnover_out, replace=False)
            idx_new_out = np.random.choice(out_pool, size=n_turnover_out, replace=False)
        elif (n_turnover_out > len(out_connectivity[i])) and (len(out_connectivity[i]) > 0):
            idx_lost_out = np.random.choice(np.arange(len(out_connectivity[i])), size=len(out_connectivity[i]), replace=False)
            idx_new_out = np.random.choice(out_pool, size=len(out_connectivity[i]), replace=False)
        for r in range(n_turnover_out):
            if (len(out_connectivity[i]) > 0):
                out_connectivity[i][idx_lost_out[r]] = idx_new_out[r]
            else:
                out_connectivity[i] = out_connectivity[i] 
        for j in range(n_turnover_in):
            in_connectivity[i][idx_lost_in[j]] = idx_new_in[j]
    return [in_connectivity, out_connectivity]
    

def interneuron_syn_loss(place_cells, out_connectivity, day_count=0):
    """ Interneuron-to-place cell synapse loss

    Parameters
    ----------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.
    out_connectivity : list
        List of interneuron-to-place cell synapses.
    day_count : int, optional
        Day count for inhibitory synapse loss. The default is 0.

    Returns
    -------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.
    out_connectivity : list
        List of updated interneuron-to-place cell synapses relfecting interneuron-to-place cell synapse loss.
    day_count : int, optional
        Day count for inhibitory synapse loss.

    """

    n_pc = place_cells['synapses'].shape[1]
    n_interneurons = int(np.ceil((n_pc/8)))
    #calculate synapse loss using quadratic equation based on values found in Schmid et al. (2010) 
    total_GABAsynapse = n_interneurons * int(np.ceil(928 * (n_pc/311500)))
    #determine timepoint of simulation
    timepoint = day_count/30
    pr_timepoint = (day_count-1)/30
    #determine percentage of synapses lost at current timepoint
    percent_loss_uptodate = -1*((-(0.0532*(timepoint**2)) - (2.2179*timepoint))/100)
    percent_loss_uptopreviousdate = -1*((-(0.0532*(pr_timepoint**2)) - (2.2179*pr_timepoint))/100)
    percent_day = percent_loss_uptodate - percent_loss_uptopreviousdate
    #update day count
    day_count += 1
    #number of synapses lost at current timepoint
    n_loss = int(np.round(percent_day * total_GABAsynapse)) 
    #pick n_loss random interneurons that will be affected 
    INs_affected = np.random.choice(np.arange(n_interneurons), size=n_loss, replace=True) #replace=True to allow one interneuron to lose multiple synapses  
    
    for i in range(n_interneurons):  
        #get utputs for each interneuron
        PC_output = out_connectivity[i]
        PC_output = [int(x) for x in PC_output]
        #AD-related synapse loss 
        
        if i in INs_affected:
            n_SYN_affected = np.count_nonzero(INs_affected == i)
            stop_point = len(PC_output) - n_SYN_affected 
            if stop_point > 0:
                PC_output = PC_output[0:stop_point] 
            else:
                PC_output = []
                print('Interneuron %i has lost all synapses'%i)
            out_connectivity[i] = PC_output
            
    return [place_cells, out_connectivity, day_count]


def interneurons_inp(place_cells, grid, winner_quantile=0.9, in_connectivity=[], out_connectivity=[]):
    """Initialises interneuron-place cell synapses and implements feedback inhibition of place cells

    Parameters
    ----------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.
    grid : array
        Defines grid cell firing rates.
    winner_quantile : float, optional
        Fraction of the maximum firing rate of any cell that a cell has to achieve to escape inhibition.. The default is 0.9.
    in_connectivity : list, optional
        List of place cell-to-interneuron synapses. The default is an empty list.
    out_connectivity : list, optional
        List of interneuron-to-place cell synapses. The default is an empty list.

    Returns
    -------
    place_cells : dict
        Dictionary with updated place cell firing rates reflecting competitive inhibition.
    in_connectivity : list, optional
        Initialised list of place cell-to-interneuron synapses on first iteration.
    out_connectivity : list, optional
        initialised list of interneuron-to-place cell synapses on first iteration.

    """
    
    #calculate PC firing rates
    y = np.dot(grid, place_cells['weights'])
    #determine number of interneurons
    n_pc = place_cells['synapses'].shape[1]
    n_interneurons = int(np.ceil((n_pc/8)))
    #initialise connectivity on first iteration 
    if len(in_connectivity) == 0:
        #number of PC inputs to each interneuron
        cell_input = int(np.ceil(728 * (n_pc/311500)))
        in_connectivity = []
        #number of projections to PC from each interneuron
        cell_output = int(np.ceil(928 * (n_pc/311500)))
        out_connectivity = []
        #set connectivity for each interneuron
        for i in range(n_interneurons):
            PC_input = np.random.choice(np.arange(n_pc), size=cell_input, replace=False)
            in_connectivity.append(list(PC_input))
            PC_output = np.random.choice(np.arange(n_pc), size=cell_output, replace=False)
            out_connectivity.append(list(PC_output))
    
    #Competitive inhibition 
    for i in range(n_interneurons):  
        #get inputs and outputs for each interneuron
        PC_input = in_connectivity[i]
        PC_input = [int(x) for x in PC_input]
        PC_output = out_connectivity[i]
        PC_output = [int(x) for x in PC_output]
        #Competitive inhibition
        for p in range(100): 
            threshold_firing = np.amax(y[p,PC_input]) * winner_quantile 
            for j in range(len(PC_output)):
                if y[p,PC_output[j]] < threshold_firing:
                    y[p,PC_output[j]] = 0.
    place_cells['firing_rates'] = y
    
    return [place_cells, in_connectivity, out_connectivity]


def scaling_function(x, scaling):
    """ Synaptic scaling function used within Hebbian and BCM learning function

    Parameters
    ----------
    x : 1D array
        Array of weights of all synapses converging onto one place cell.
    scaling : float
        Expected sum of synaptic weights converging onto a place cell.

    Returns
    -------
    ans : 1D array
        Array of scaled weights of all synapses converging onto one place cell.

    """

    if np.sum(x) >0:
        ans = (scaling/np.sum(x)) *x
    else:
        ans = x
    return ans

def update_hebbian(place_cells, grid, learning_rate=0.001, scaling=149.1):
    """ Hebbian learning rule for grid cell-to-place cell synapses

    Parameters
    ----------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.
    grid : array
        Defines grid cell firing rates.
    learning_rate : float, optional
        Hebbian learning rate. The default is 1e-3.
    scaling : float, optional
        Expected sum of synaptic weights converging onto a place cell. The default is 149.1, 
        expected in a network with 1,200 grid cell-to-place cell synapses.

    Returns
    -------
    place_cells : dict
        Dictionary with updated place cell synaptic weights.

    """

    y = place_cells['firing_rates']
    dw = np.dot(grid.T, y) 
    dw = place_cells['synapses'] * dw
    weight_change = (learning_rate*dw)
    place_cells['weights'] = place_cells['weights'] + weight_change

    #synaptic scaling 
    place_cells['weights'] = np.apply_along_axis(scaling_function,0, place_cells['weights'], scaling=scaling)
    #delete any potential negative weights 
    place_cells['weights'] = (place_cells['weights'] >0.)*place_cells['weights']
    
    #optional upper bounds for synaptic weights 
    #upperlimit = np.where(place_cells['weights'] > 2.)
    #place_cells['weights'][upperlimit] = 2.
    
    return place_cells

def update_bcm(place_cells, grid, scaling=149.1, limit=2):
    """ BCM learning rule for grid cell-to-place cell synapses

    Parameters
    ----------
    place_cells : dict
        Dictionary with place cell synapses, synaptic weights and firing rates arrays.
    grid : array
        Defines grid cell firing rates.
    scaling : float, optional
        Expected sum of synaptic weights converging onto a place cell. The default is 149.1, 
        expected in a network with 1,200 grid cell-to-place cell synapses.
    limit : int or float, optional
        Limit l for y*(y-T) term. The default is 2.

    Returns
    -------
    place_cells : dict
        Dictionary with updated place cell synaptic weights.

    """ 

    y = place_cells['firing_rates']
    #positive constant F_0 
    F_0 = 50
    #get dynamic threshold for each place cell 
    for c in range(y.shape[1]):
        F_mean = np.mean(y[:,c])
        T = ((F_mean/F_0)**2)*F_mean
        y[:,c] = y[:,c]*(y[:,c]-T)       
        for i in range(len(y[:,c])):
            if y[i,c] > limit:
                y[i,c] = limit
            elif y[i,c] < (-limit):
                y[i,c] = -limit
    #update synaptic weights 
    dw = np.dot(grid.T,y)
    dw = place_cells['synapses'] * dw 
    place_cells['weights'] = place_cells['weights'] + dw 
    #prevent negative weights 
    place_cells['weights'] = (place_cells['weights'] >= 0.)*place_cells['weights'] 
    #synaptic scaling
    place_cells['weights'] = np.apply_along_axis(scaling_function,0, place_cells['weights'], scaling=scaling)
    return place_cells
    

def rle(response_array):
    """ Run length encoding function to get place field centroids in centroid_fun. 

    Parameters
    ----------
    response_array : 1D numpy array or list
        True/False vector array of place cell firing with True where place cell's firing rate 
        is above 50% of the maximum firing rate.

    Returns
    -------
    place_cells : dict
        Dictionary with updated place cell synaptic weights.
    
    """

    values = np.asarray(response_array) 
    N = len(values)
    if N == 0:
        return (None, None, None)
    else:
        y = np.array(values[1:] != values[:-1])     
        i = np.append(np.where(y), N - 1)   
        run_length = np.diff(np.append(-1, i))       
        start_pos = np.cumsum(np.append(0, run_length))[:-1] 
    return(run_length, start_pos, values[i])

def centroid_fun(response_cell, min_field_width=5):
    """ Function used within get_centroids to find place field centroids

    Parameters
    ----------
    response_cell : 1D numpy array 
        Array of firing rates of the a place cell.
    min_field_width : int, optional
        Minimum length of place fields. Default is 5.

    Returns
    -------
    centroid: int
        Position of centroid of the cell's place field along the track. 0 if no place field detected.
        
  """
    maximum = np.max(response_cell)
    runs = rle(response_cell >(maximum*0.5)) #gives True/False vector and runs rle on it
    long_runs = (runs[0] > min_field_width) & (runs[2] == True) & (runs[0] < 50)
    if (np.sum(long_runs) == 1):
        centroid = np.cumsum(runs[0])[long_runs] - runs[0][long_runs]/2 
    else:
        centroid = 0
        #no PF longer than 5 positions has central point at position 0 
    return centroid
 
def get_centroids(response, min_field_width=5):
    """ Function to apply centroid_fun to a 2D array of place cell firing rates.

    Parameters
    ----------
    response : 2D array
        Firing rates of place cells
    min_field_width : int, optional
        Minimum length of place fields. Default is 5.

    Returns
    -------
    centroids : 1D array
        Array of place field centroid values and 0's'

    """

    centroids = np.apply_along_axis(centroid_fun,0, response)
    return centroids
    
def centroid_widths(response, min_field_width=5):
    """ Function to get place field widths

    Parameters
    ----------
    response : 2D array
        Firing rates of place cells
    min_field_width : int, optional
        Minimum length of place fields. The default is 5.

    Returns
    -------
    median_width : float
        Median place field width.
    mean_width : float
        Median place field width.
    width_std : float
        Standard deviation of place field widths.

    """

    widths = []
    for i in range(response.shape[1]):
        maximum = np.max(response[:,i])
        runs = rle(response[:,i] >(maximum*0.5)) #gives True/False vector and runs rle on it
        long_runs = (runs[0] > min_field_width) & (runs[0] < 50) & (runs[2] == True)
        if (np.sum(long_runs) == 1):
            width = runs[0][long_runs]
            widths.append(width)
    median_width = np.median(widths)
    mean_width = np.mean(widths)
    width_std = np.std(widths)
    return (median_width, mean_width, width_std)

def get_place_field_properties(place_cell_list, samples):
    """ Function to get main place cell properties from simulation, including number of 
        place cells and place field widths

    Parameters
    ----------
    place_cell_list : dict
        Dictionary of place cell array for each sampled day in the simulation.
    samples : 1D array or list
        Timepoints at which to determine place cell properties.

    Returns
    -------
    tpcs : list
        Total number of place cells on each day given in samples.
    pf_width : list
        Median place field width on each day given in samples.
    pf_width_mean : list
        Mean place field width on each day given in samples.
    pf_width_std : list
        Standard deviation of place field width on each day given in samples.

    """
    # new working dictionary with place cell firing from sim output tuple
    pcs = {} 
    for d in samples:
        pcs[d] = np.array(place_cell_list[d]['firing_rates'])
    #get PF width
    [avg_width_0, mean_width_0, width_std_0] = centroid_widths(pcs[samples[0]])
    #get centroids and locations of centroids on day 0
    centroids_day0 = get_centroids(pcs[samples[0]])
    if len(centroids_day0) == 1:
        centroids_day0 = centroids_day0[0] #avoid array in array problem 
    centroids_loc_day0 = [i for i,e in enumerate(centroids_day0) if e!= 0]
    
    #Day 0 
    tpcs = []
    tpcs.append(len(centroids_loc_day0))
    pf_width = [avg_width_0]
    pf_width_mean = [mean_width_0]
    pf_width_std = [width_std_0]
    
    for i in samples[1:]:
        [median_width_i, mean_width_i, width_std_i] = centroid_widths(pcs[i])
        pf_width.append(median_width_i)
        pf_width_mean.append(mean_width_i)
        pf_width_std.append(width_std_i) 
        centroids_day_i = get_centroids(pcs[i])         
        if len(centroids_day_i) == 1:
             centroids_day_i = centroids_day_i[0]
        centroids_loc_day_i = [i for i,e in enumerate(centroids_day_i) if e!= 0]
        tpcs.append(len(centroids_loc_day_i))

    return (tpcs, pf_width, pf_width_mean, pf_width_std)

def get_activity_distribution(sim_run, samples):
    """ Get activity distribution of place cells (see Figure 6)
    
    Parameters
    ----------
    sim_run : dict
        Dictionary of place cell array for each sampled day in the simulation.
    samples : 1D array or list
        Timepoints at which to determine place cell properties.

    Returns
    -------
    cell_id: 2D array
        Gives activity level of each cell for each position.

    """    
    n_pc = np.array(sim_run[samples[0]]['firing_rates']).shape[1]
    
    cell_id = np.zeros((n_pc, len(samples))) # should contain activity group for each cell at each time point
    #id's saved as: 0: silent, 1:rare, 2:intermediate, 3:high
    high_cutoff= 200
    low_cutoff = 50
    for c, i in enumerate(samples):
        firing = np.array(sim_run[i]['firing_rates'])        
        for j in range(n_pc):
            s = np.sum(firing[:,j])
            if s>0:
                if s> high_cutoff: #highly active
                    cell_id[j,c] = 3
                elif s<low_cutoff: #rarely active
                    cell_id[j,c] = 1
                else: #intermediately active
                    cell_id[j,c] = 2
            else: #silent
                cell_id[j,c] = 0

    return [cell_id] 

def get_place_fields(day_firing_rates, min_field_width=5):
    """ Get array with firing rates in place field positions only (see Fig 1A)

    Parameters
    ----------
    day_firing_rates : 2D array
        Array of firing rates of place cells on one simulation day.
    min_field_width : int, optional
        Minimum place field width. The default is 5.

    Returns
    -------
    pfs.T : 2D array
        Array with firing rates of cells at place field positions, 0 otherwise.

    """
    firing_rates = np.array(day_firing_rates)
    pfs = np.zeros(firing_rates.shape)
    
    for col in range(firing_rates.shape[1]):
        maximum = np.max(firing_rates[:,col])
        runs = rle((firing_rates[:,col] > (maximum*0.8)) )#& (firing_rates[:,col] > 1))
        long_runs = (runs[0] > min_field_width) & (runs[2] == True)
        if (np.sum(long_runs) == 1):
            start = int(runs[1][long_runs])
            end = int(np.cumsum(runs[0])[long_runs])
            pfs[start:(end+1), col] = firing_rates[start:(end+1),col]  #
    return pfs.T


def get_new_pc_indices(sim_run, samples):
    """ Get number of new place cells

    Parameters
    ----------
    sim_run : dict
        Dictionary of place cell array for each sampled day in the simulation.
    samples : 1D array or list
        Timepoints at which to determine place cell properties.

    Returns
    -------
    new_pc_indices : list
        Number of new place cells since last sample for all timepoints in samples.

    """
    pc_indices = []
    for i, d in enumerate(samples):  
        pfs = get_place_fields(sim_run[d]['firing_rates'])
        pc_idx = []
        for row in range(pfs.shape[0]):
            o = np.where(pfs[row,:] >0)
            if len(o[0]) > 0: 
                pc_idx.append(row)
        pc_indices.append(pc_idx)

    new_pc_indices = [] 
    for i in range(len(samples)):
        if i==0:
            new_pc_indices.append(len(pc_indices[0]))
        else:
            new_pcs=[]
            for j in pc_indices[i]:
                if j not in pc_indices[i-1]:
                    if j not in pc_indices[0]:
                        new_pcs.append(j)
            new_pc_indices.append(len(new_pcs))
    return new_pc_indices

def get_rpcs(sim_run, samples):
    """ Get recurring place cells 

    Parameters
    ----------
    sim_run : dict
        Dictionary of place cell array for each sampled day in the simulation.
    samples : 1D array or list
        Timepoints at which to determine place cell recurrence with samples[1] as reference
        
    Returns
    -------
    rpcs : list
        Number of recurring place cells with less than 5 units in centroid drift between any two timepoints.

    """

    pcs = {}
    for d in samples:
        pcs[d] = np.array(sim_run[d]['firing_rates'])
    centroids_day0 = get_centroids(pcs[samples[0]])
    if len(centroids_day0) == 1:
        centroids_day0 = centroids_day0[0] #avoid array in array problem 
    centroids_loc_day0 = [i for i,e in enumerate(centroids_day0) if e!= 0]
    rpcs = [len(centroids_loc_day0)]
    
    for i in samples[1:]: 
        centroids_day_i = get_centroids(pcs[i])         
        if len(centroids_day_i) == 1:
             centroids_day_i = centroids_day_i[0]
        centroids_loc_day_i = [i for i,e in enumerate(centroids_day_i) if e!= 0]
        
        #safe first centroids after day 0 as reference 
        if i==samples[1]:
            ref_loc_centroids = centroids_loc_day_i
            ref_centroids_position = centroids_day_i

        rc_loc = [x for x in centroids_loc_day_i if x in ref_loc_centroids]
        pos_change = np.array(abs(centroids_day_i[rc_loc] - ref_centroids_position[rc_loc]))
        acceptchange = np.where(pos_change <=5)
        acceptchange = [x for x in acceptchange[0]]
        ref_centroids_position = centroids_day_i

        rc_loc = [e for i,e in enumerate(rc_loc) if i in acceptchange]
        rpcs.append(len(rc_loc))
    return (rpcs)

def get_recurrence_odds(sim_run, samples, rpcs): 
    """ Get probability for recurrence of a place cell or active cell between timepoints

    Parameters
    ----------
    sim_run : dict
        Dictionary of place cell array for each sampled day in the simulation.
    samples : 1D array or list
        Timepoints at which to determine place cell recurrence with samples[1] as reference
    rpcs : list
        Number of recurring place cells with less than 5 units in centroid drift between any two timepoints.
        Output of get_rpcs function.

    Returns
    -------
    active_probs : list
        Probability of recurrence of an active cell between two timepoints.
    recurrence_probs : list
        Probability of recurrence of a place cell between two timepoints.

    """
    n_pc = np.array(sim_run[samples[0]]['firing_rates']).shape[1]
    cell_id = np.zeros((n_pc, len(samples)))

    for c, i in enumerate(samples):
            firing = np.array(sim_run[i]['firing_rates'])
            for j in range(n_pc):
                s = np.sum(firing[:,j])
                if s>0:
                    cell_id[j,c] = 1
                elif s==0:
                    cell_id[j,c] = 0

    cell_activities = pd.DataFrame(cell_id, columns=samples)
    active_probs = []
    for i in range(1,len(samples)):
        overlap = pd.crosstab(cell_activities[samples[1]],cell_activities[samples[i]])[1][1]
        final_per = overlap/ pd.crosstab(cell_activities[samples[1]],cell_activities[samples[1]])[1][1]
        active_probs.append(final_per)

    recurrence_probs = []
    for i in range(1,len(samples)):
        recurrence_probs.append(rpcs[i]/rpcs[1])
    
    return (active_probs, recurrence_probs)

#%% Run simulation

#Load grid cell firing from grid_cells-2d with activities of 10,000 grid cell
grid = np.load("grid_cells-2d.npy")
grid = grid.T
grid = grid[:,:5000]

#Load synaptic strength pool
with open("syn_str_pool.csv", newline='') as csvfile: 
    reader = csv.reader(csvfile, delimiter=' ')
    syn_str_pool = list(reader)
synaptic_strength_pool = []
for e in syn_str_pool:
    synaptic_strength_pool.append(float(e[0]))
 
def simulate_place_cells(grid, samples, learning_rate=1e-3, gc_syn_loss=False, inh_syn_loss=False, n_pc=1000):
    """ Simulate place cell activity

    Parameters
    ----------
    grid : 2D array
        Grid cell activity array.
    learning_rate : float, optional
        Learning rate for Hebbian learning.
    samples : 1D array or list
        Timepoints at which place cell activity samples are stored. Maximum defines length of simulation.
    gc_syn_loss : True/False, optional
        Defines whether AD-related grid cell-to-place cell synaptic loss is implemented. The default is False.
    inh_syn_loss : True/False, optional
        Defines whether AD-related interneuron-to-place cell synaptic loss is implemented. The default is False.
    n_pc : int, optional
        Number of pyramidal cells in simulation. The default is 1000.

    Returns
    -------
    place_cell_list : Dict
        Dictionary of sample timepoints with place cell synapses, syanptic weights and firing rate arrays.

    """
    place_cell_list = {} 

    n_grids = grid.shape[1] #number of grid cells 
    winner_quantile=0.90 #for competitive inhibition

    #get the expected sum of synaptic weight for each place cell 
    E=[]
    for i in range(10000):
        n_syn = int(np.ceil(1200*(n_pc/311500)))
        e=np.random.choice(synaptic_strength_pool, n_syn, replace=True)
        E.append(np.sum(e))
    scaling=np.mean(E)

    #initialize variables
    place_cells = initialize_place_cells(n_pc, n_grids)
    initial_in_connectivity = []
    initial_out_connectivity= []
    day_count_gc = 1 #start at day 1
    day_count = 1 #start inhibitory synapse loss at day 1
    
    #get initial firing rate and initialise interneuron-PC architecture 
    [place_cells, in_connectivity, out_connectivity] = interneurons_inp(place_cells, grid, winner_quantile, in_connectivity=initial_in_connectivity, out_connectivity=initial_out_connectivity) 
    #can either use Hebbian learning (update_hebbian) or BCM learning (update_bcm) here
    #place_cells = update_hebbian(place_cells,grid,learning_rate)
    place_cells = update_bcm(place_cells, grid, scaling=scaling, limit=200)
    [place_cells, in_connectivity, out_connectivity] = interneurons_inp(place_cells, grid, winner_quantile=winner_quantile, in_connectivity=in_connectivity, out_connectivity=out_connectivity) 

    #place_cells data mapped to tuples and saved 
    pc = {}
    pc['synapses'] = tuple(map(tuple, place_cells['synapses']))
    pc['weights'] = tuple(map(tuple, place_cells['weights']))
    pc['firing_rates'] = tuple(map(tuple, place_cells['firing_rates']))
    place_cell_list[0] = pc.copy()
    
    #Export data for re-import if there are memory issues due to length of simulation
    #np.savez_compressed('sim_0.npz', synapses=place_cells['synapses'], weights=place_cells['weights'], firing=place_cells['firing_rates'])
    
    #day 1 and beyond 
    for d in range(1,np.max(samples)+1):
        place_cells = turnover(place_cells) 
        [in_connectivity, out_connectivity] = interneuron_synapse_turnover(in_connectivity, out_connectivity) #interneuron synapse turnover 
        #AD related synapse loss
        if (gc_syn_loss==True): 
            [place_cells, day_count_gc] = synapse_loss(place_cells, day_count_gc)
        if (inh_syn_loss==True):
            [place_cells, out_connectivity, day_count] = interneuron_syn_loss(place_cells=place_cells, out_connectivity=out_connectivity, day_count=day_count)
        
        [place_cells, in_connectivity, out_connectivity] = interneurons_inp(place_cells, grid,  winner_quantile=winner_quantile, in_connectivity=in_connectivity, out_connectivity=out_connectivity)
        #place_cells = update_hebbian(place_cells, grid, learning_rate)
        place_cells = update_bcm(place_cells, grid, scaling=scaling, limit=200)
        [place_cells, in_connectivity, out_connectivity] = interneurons_inp(place_cells, grid, winner_quantile=winner_quantile, in_connectivity=in_connectivity, out_connectivity=out_connectivity)
        
        #save data 
        if d in samples:
            print('Days elapsed: %i'%d)
            #np.savez_compressed('sim_%i'%d, synapses=place_cells['synapses'], weights=place_cells['weights'], firing=place_cells['firing_rates'])
            pc['synapses'] = tuple(map(tuple, place_cells['synapses']))
            pc['weights'] = tuple(map(tuple, place_cells['weights']))
            pc['firing_rates'] = tuple(map(tuple, place_cells['firing_rates']))
            place_cell_list[d] = pc.copy() #save place cell data
    return place_cell_list

""" Set parameters:
    eta: learning rate for Hebbian learning
    n_pc: number of place cells
    gc_syn_loss: True if grid cell-to-place cell synaptic loss implemented
    inh_syn_loss: True if interneuron-to-place cell synaptic loss implemented
    samples: 'Days' on which to save simulation data 
    
    """
eta = 1e-3 
n_pc = int(np.ceil(15575*0.5)) 
gc_syn_loss= False
inh_syn_loss=False
samples = np.arange(0,31,5)

sim_run = simulate_place_cells(grid, samples=samples, learning_rate=eta, gc_syn_loss=gc_syn_loss, inh_syn_loss = inh_syn_loss, n_pc=n_pc)

#%% Analyse place cell properties

(tpcs, pf_width, pf_width_mean, pf_width_std) = get_place_field_properties(sim_run, samples)
cell_id = get_activity_distribution(sim_run, samples)
npcs = get_new_pc_indices(sim_run, samples)

recurrence_samples = np.arange(0,31,5)
rpcs = get_rpcs(sim_run, recurrence_samples)
[active_probs, recurr_probs] = get_recurrence_odds(sim_run, recurrence_samples, rpcs)

#%% Demo graphs 

fig, (ax, ax1, ax2) = plt.subplots(3,1, figsize=(5,25))

ax.errorbar(samples, tpcs, c='C0', label = 'Place cells')
ax.errorbar(samples, npcs, c='C0', ls='--', label = 'New place cells')
ax.set_title('Number of place cells and new place cells')
ax.set_ylabel('Number of cells')
ax.locator_params(axis='y', nbins=6)
ax.tick_params(axis='x', direction='out', left='on', labelleft='on')
ax.tick_params(axis='y',direction='out', left='on',labelleft='on')
ax.legend(facecolor='1', edgecolor='1', loc='upper right')
ax.set_ylim(bottom=0)

ax1.errorbar(samples, pf_width_mean, color='C0')
ax1.set_title('Mean place field width')
ax1.set_ylabel('Mean width (cm)')
ax1.locator_params(axis='y', nbins=6)
ax1.set_ylim(bottom=0, top=20)
ax1.tick_params(axis = 'both', which = 'major')

ax2.errorbar(recurrence_samples[:-1], recurr_probs, label = 'Place cells', ls='-', marker='.', color='C0')
ax2.errorbar(recurrence_samples[:-1], active_probs, label='Active cells', ls='--', marker='.', color='C0', alpha=0.4)
ax2.set_title('Recurrence probability')
ax2.set_ylim(0,1)
ax2.set_ylabel('Probability of recurrence')
ax2.legend(facecolor='1', edgecolor='1', loc='upper right')
ax2.locator_params(axis='y', nbins=8)

for axis in [ax,ax1,ax2]:
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.set_xlabel('Time from first session (days)')
    
fig.subplots_adjust(hspace=0.8)


