#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 12:26:43 2021

@author: brian
"""


from gittislab import table_wrappers, signals, behavior, dataloc, ethovision_tools, plots, model
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from matplotlib import pyplot as plt
import pdb
from itertools import compress
import seaborn as sns
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
    savepath = Path('F:\\Users\\Gittis\\Dropbox\\Manuscripts\\Isett_Gittis_2021\\')
    basepath = Path('F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\')
else:
    savepath = Path('/home/brian/Dropbox/Manuscripts/Isett_Gittis_2021/')
    basepath = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/')
    
fig_name='Figure 1_a2a_chr2_str'

newbase = savepath.joinpath(fig_name).joinpath('data').joinpath('raw')

# %%
# inc = [x + ['AG3233_5'] for x in inc]
ethovision_tools.raw_csv_to_preprocessed_csv(newbase,inc,exc,
                                             force_replace=True,win=10)


# %%

pns=dataloc.raw_csv(newbase,inc[0],exc[0])
if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=True
for pn in pns:
    df,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_openloop_day(df,meta,
                            save=saveit,
                            close=closeit,
                            save_dir=pn.parent)

# %% 


analysis='Str_A2a_ChR2_1mw'
behavior_str='10x'

fn = '%s_openloop_data' % analysis
qans = ['AG3233_5','AG3233_4', 'AG3488_7',]
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,
                                                          behavior_str=behavior_str)
summary = ethovision_tools.meta_sum_csv(newbase,inc,exc)

data = behavior.open_loop_summary_collect(newbase,inc,exc,
                                          update_rear=True,
                                          stim_analyze_dur=10)

smooth_amnt= [33, 33]
fig,stats,axs =plots.plot_openloop_mouse_summary(data,smooth_amnt=smooth_amnt,method=[10,1])

 # %% zone test
 

analysis='Str_A2a_ChR2_1mw'
behavior_str='zone_'
fn = '%s_rtpp_data' % analysis

inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,
                                                          behavior_str=behavior_str)

summary = ethovision_tools.meta_sum_csv(newbase,inc,exc)
summary = summary.sort_values('anid').reset_index()

data = behavior.zone_rtpp_summary_collect(newbase,inc,exc,
                                          update_rear=True,
                                          stim_analyze_dur='mean',
                                          zone_analyze_dur=10*60) #Analyze first 10 minutes 


# %% 
u_mice = np.unique(data.anid)
if len(data) > len(u_mice):
    print('Warning! Duplicate mice detected. \nZone 1 experiment used if 1 & 2 found.')
    for mouse in u_mice:
        if np.sum(data.loc[:,'anid'] == mouse) > 1:
            exc_row = np.argwhere( (data.loc[:,'anid'].values == mouse) 
                              & (data.loc[:,'proto'].values == 'zone_2')).flatten()
            data.drop(axis=0,index=exc_row, inplace=True)
            summary.drop(axis=0,index=exc_row,inplace=True)
else: 
    print('Data as %d unique mice.' % len(data))
data.reset_index(inplace=True)
