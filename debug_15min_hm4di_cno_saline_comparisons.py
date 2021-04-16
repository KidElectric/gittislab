#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:27:21 2021

@author: brian
"""


from gittislab import signals, behavior, dataloc, ethovision_tools, plots, table_wrappers
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import pdb
from itertools import compress
import seaborn as sns

ex0=['exclude','Bad','GPe','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','hm4di','Str','A2A','Ai32']]
# inc=[['AG','Str','D2_D1','ChR2_hM3Dq','10x10_15mW']]
make_preproc = False
exc=[ex0]
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
        
    basepath = 'F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\'
else:
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

# %% Plot 15min free running mouse summary:
ex0=['10hz','exclude','Bad','bad',
     'Broken', 'grooming','Exclude','Other XLS']
exc=[ex0]
conds = ['saline','cno']
keep={}
for cond in conds:
    inc=[['AG','hm4di','Str','A2A','Ai32','15min',cond]]
    data = behavior.free_running_summary_collect(basepath,inc,exc)
    keep[cond]=data
    plots.plot_freerunning_mouse_summary(data,)

# %% Plot both conditions:
plots.plot_freerunning_cond_comparison(keep,save=False,close=False)

# %% Calculate openloop mouse summary across conditions:
conds = ['saline','cno']
ex0=['exclude','Bad','GPe','bad','Broken',
     'Exclude','Other XLS','10hz','15min','3mW']
exc=[ex0]
keep={}
for cond in conds:
    inc=[['AG','hm4di','Str','A2A','Ai32','10x10',cond]]
    data = behavior.open_loop_summary_collect(basepath,inc,exc)
    keep[cond]=data
    
# %% Plot
plots.plot_openloop_cond_comparison(keep,save=False,close=False)
    
# %% Calculate a mouse-wise differeince in percent time mobile base vs. stim:
anids=np.unique(keep[conds[0]]['anid'])
#Assume for now that rows have been sorted already by animal ID and are all matching
ds=[]
for cond in conds:
    dat=np.stack(keep[cond]['per_mobile'])
    ds.append((dat[:,0] - dat[:,1]))
ds=np.stack(ds).transpose()
ds=np.stack([x/ds[:,0]*100 for x in ds.T]).T
f,a,h=plots.connected_lines(conds,ds)

for i,anid in zip(h,anids):
    i.set_label(anid)
plt.ylabel('Normalized \Delta %Time Mobile')
plt.legend()
        