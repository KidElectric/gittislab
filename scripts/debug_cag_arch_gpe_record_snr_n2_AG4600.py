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

n_exc=[]
n_inh=[]
n_ns=[]
exps = np.unique(df['exp_name'])
for exp in exps:
    use= [exp in x for x in df['exp_name']]
    n_exc.append(sum(exc & use))
    n_inh.append(sum(inh & use))
    n_ns.append(sum(use & ~sig))
    percent_exc = sum(exc & use) / sum(use) * 100
    percent_inh = sum(inh & use) / sum(use) * 100 
    print('%2.1f%% exc, %2.1f%% inh, ' % (percent_exc,percent_inh))

b=np.array((n_exc,n_inh,n_ns))
fig, ax = plt.subplots()

labels = [x[0] for x in exps]
tot = np.array(n_inh) + np.array(n_ns)
width = 0.35       # the width of the bars: can also be len(x) sequence
ax.bar(labels, n_ns, width, label = 'n.s.',facecolor='k')
ax.bar(labels, n_inh, width, label = 'inh.',facecolor='b',bottom=n_ns)
ax.bar(labels, n_exc, width, label = 'exc.',facecolor='r',bottom = tot)
plt.ylabel('Number Units')
plt.xlabel('Experiment')
percent_exc=sum(b[0,:]) / len(sig) * 100
percent_inh=sum(b[1,:]) / len(sig) * 100
plt.title('%2.1f%% exc, %2.1f%% inh' % (percent_exc,percent_inh))
ax.legend()