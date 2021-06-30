# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:18:56 2021

@author: Brian
"""


from gittislab import signals, behavior, dataloc, ethovision_tools, plots, model
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import pdb
from itertools import compress


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
    
# %% RUN BEFORE DLC:
ex0=['exclude','Bad','Str','bad','Broken', 'grooming',
     'Exclude','Other XLS']
exc=[ex0]
#inc=[['AG','ChR2_hM3Dq','Str','15mW_cno','D2_D1',]]
#inc=[['AG','hm4di','Str','A2A','Ai32','10x10','3mW']]
# inc=[['AG','Str','A2A','Ai32','zone_2_0p5mw',]]
# inc=[['AG','Str','A2A','Ai32','50x2_multi_mW',]]

ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = make_preproc)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,force_replace=False,win=10)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% RUN AFTER DLC:
# ex0=['exclude','Bad','Str','bad','Broken', 'grooming','Exclude','Other XLS']
# exc=[ex0]
# inc=[['AG','hm4di','Str','A2A','Ai32','10x10','cno_10hz']]
# inc=[['AG','hm4di','Str','A2A','Ai32','10x10','3mW']]
# inc=[['AG','10x10','Str','A2A','Ai32','AG5362_3']]
# inc=[['AG','Str','A2A','Ai32','50x2_hm4di_sal',]]
# inc=[['AG','Str','A2A','Ai32','10x10','0p25mw']]
# inc=[['AG','GPe','CAG','Arch','10x30',]]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = False)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,
                                             force_replace=True,win=10)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))
# %% Plot zone day
inc=[['AG','Str','A2A','Ai32','zone_2','_0p5mw']]
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=False
for pn in pns:
    raw,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_zone_day(raw,meta,save=saveit, close=closeit)
    
# %% Plot zone day summary:   
inc=[['AG','Str','A2A','Ai32','zone','_0p5mw']]
# data = behavior.zone_rtpp_summary_collect(basepath,inc,exc)
plots.plot_zone_mouse_summary(data,example_mouse=1)
# %% Plot 10x10 / openloop days:
ex0=['exclude','Bad','GPe','bad','Broken','15min', '10hz','grooming','Exclude','Other XLS']
# exc=[ex0]
inc=[['AG','Str','A2A','Ai32','10x10','0p25mw']]
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=False
for pn in pns:
    df,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_openloop_day(df,meta,save=saveit, close=closeit)

# %% Plot free-running / unstructured openfield data:
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=True
for pn in pns:
    df,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc')
    plots.plot_freerunning_day(df,meta,save=saveit, close=closeit)
    
# %% Plot openloop mouse summary:
ex0=['exclude','Bad','GPe','bad',\
     'Broken','15min','10hz', 'Exclude','Other XLS','AG3233_5','AG3233_4',\
         'AG3488_7']
# exc=[ex0]
# inc=[['AG','hm4di','Str','A2A','Ai32','saline','10x10','3mW']]
# inc=[['AG','Str','A2A','Ai32','10x10','0p25mw']]
# inc=[['AG','Str','A2A','Ai32','zone_1_0p5mw',]]
# inc=[['AG','Str','A2A','Ai32','zone','_0p5mw']]
# inc=[['AG','GPe','CAG','Arch','10x30',]]
inc=[['AG','Str','A2A','ChR2','10x30','Bilateral'],['AG','Str','A2A','Ai32','10x30','Bilateral']]
exc=[ex0,ex0]
data = behavior.open_loop_summary_collect(basepath,inc,exc)
fig=plots.plot_openloop_mouse_summary(data)

# %% Plot openloop mouse summary across conditions:
conds = ['saline','cno']
ex0=['exclude','Bad','GPe','bad','Broken',
     'Exclude','Other XLS','10hz','15min',]
exc=[ex0]
keep={}
for cond in conds:
    inc=[['AG','hm4di','Str','A2A','Ai32','10x10','3mW',cond]]
    data = behavior.open_loop_summary_collect(basepath,inc,exc)
    keep[cond]=data
# %%
plots.plot_openloop_cond_comparison(keep,save=False,close=False)

# %%
fig=plots.plot_openloop_mouse_summary(data)

# %% Plot 15min free running mouse summary:
ex0=['10hz','exclude','Bad','bad',
     'Broken', 'grooming','Exclude','Other XLS']
exc=[ex0]
# inc=[['AG','hm4di','Str','A2A','Ai32','15min','cno']]
conds = ['saline','cno']
keep=[]
for cond in conds:
    inc=[['AG','hm4di','Str','A2A','Ai32','15min',cond]]
    data = behavior.free_running_summary_collect(basepath,inc,exc)
    keep.append(data)
    plots.plot_freerunning_mouse_summary(data,)
    
#%%
inc=[['AG','hm4di','Str','A2A','Ai32','10x10','saline']]
data = behavior.open_loop_summary_collect(basepath,inc,exc)
fig=plots.plot_openloop_mouse_summary(data)



# %% 
