# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:47:37 2020

@author: diego
"""

# Importing packages

import pandas as pd
import numpy as np

#%% Convert samples to an empirical distribution

def delay_distribution(delay):

    p_delay = delay.value_counts().sort_index()
    new_range = np.arange(0, p_delay.index.max()+1)
    p_delay = p_delay.reindex(new_range, fill_value=0)
    p_delay /= p_delay.sum()
    
    return p_delay

#%% Importing cases for Guatemala

def confirmed_to_onset(confirmed, p_delay):

    assert not confirmed.isna().any()
    len(p_delay)
    # Reverse cases so that we convolve into the past
    convolved = np.convolve(confirmed[::-1].values, p_delay)
    
    # Calculate the new date range
    dr = pd.date_range(end=confirmed.index[-1],
                       periods=len(convolved))

    # Flip the values and assign the date range
    onset = pd.Series(np.flip(convolved), index=dr)
    
    return onset

#%% Adjusting for Right Censoring

def adjust_onset_for_right_censorship(onset, p_delay):
    
    cumulative_p_delay = p_delay.cumsum()
    
    # Calculate the additional ones needed so shapes match
    ones_needed = len(onset) - len(cumulative_p_delay)
    padding_shape = (0, ones_needed)
    
    # Add ones and flip back
    cumulative_p_delay = np.pad(
            cumulative_p_delay,
            padding_shape,
            'constant',
            constant_values=(0,1))
    
    cumulative_p_delay = np.flip(cumulative_p_delay)
    
    # Adjusts observed onset values to expected terminal onset values
    adjusted = onset / cumulative_p_delay
    
    return adjusted, cumulative_p_delay

