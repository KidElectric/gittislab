# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:18:56 2021

@author: Brian
"""


from gittislab import signals, behavior, dataloc, ethovision_tools, plots
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import pdb
from itertools import compress


ex0=['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','hm4di','Str','A2A','Ai32']]
make_preproc = True
exc=[ex0]
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
        
    basepath = 'F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\'
else:
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
    
# %% RUN BEFORE DLC:
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = make_preproc)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,force_replace=False,win=10)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% Plot 10x10 days:
inc=[['AG','hm4di','Str','A2A','Ai32','10x10']]
pns=dataloc.raw_csv(basepath,inc[0],ex0)
for pn in pns:
    df,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_openloop_day(df,meta,save=True, close=True)

# %% Plot mouse summary:
inc=[['AG','hm4di','Str','A2A','Ai32','10x10','cno']]
data = behavior.open_loop_summary_collect(basepath,inc,exc)
fig=plots.plot_openloop_mouse_summary(data)

#%%
inc=[['AG','hm4di','Str','A2A','Ai32','10x10','saline']]
data = behavior.open_loop_summary_collect(basepath,inc,exc)
fig=plots.plot_openloop_mouse_summary(data)

# %% RUN AFTER DLC:
ex0=['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS','mW']
inc=[['AG','A2A','Ai32','10x10']]
make_preproc = True
exc=[ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = make_preproc)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))