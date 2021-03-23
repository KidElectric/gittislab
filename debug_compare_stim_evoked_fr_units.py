#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 14:30:36 2021

@author: brian
"""

import os
import sys
import statistics
from scipy import stats
sys.path.append('/home/brian/Dropbox/Python')
from gittislab import mat_file
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path

# %% Combine units from mulitple analysis files:
fn='spike_stim_analysis.mat'
pns=['/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/SNr/Naive/CAG/Arch/AG6700_3_BI030221',
     '/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/SNr/Naive/CAG/Arch/AG6700_5_BI022421',]

df=pd.DataFrame()
for pn in pns:
    p=Path(pn).joinpath(fn)
    dat=mat_file.load(p)
    newdf=mat_file.array_of_arrays_to_flat_df(dat['data'])
    df=pd.concat((df,newdf))

df.reset_index(inplace = True)

#%% Calculate percentage of exc. inh. and unchaged units:
alpha = 0.05
sig = df['mod_p'] < alpha
fr_exc = df['fr_change'] > 0

exc = sig & fr_exc
inh = sig & ~fr_exc
ns = ~sig

exps = np.unique(df['exp_name'])
for exp in exps:
    use= [exp in x for x in df['exp_name']]
    

    