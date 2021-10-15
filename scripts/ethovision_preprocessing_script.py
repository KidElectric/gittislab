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
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
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

# inc=[['AG','A2A','Ai32','GPe','10x']]

# inc = [['AG','D1','DTR','Str','10x']]

analysis='Str_D1_EYFP'
behavior_str = 'zone_'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,
                                                          behavior_str=behavior_str)

basepath='/home/brian/Dropbox/Manuscripts/Isett_Gittis_2021/Figure 5_D1_arch/'
# inc=[['Str','D1','EYFP','Unilateral','Left'],
#      ['Str','D1','EYFP','Unilateral','Right']]
# inc=[[]]
exc=[[],[]]
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,
                                  force_replace=False,
                                  win=10,)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,
                                             force_replace=False,
                                             win=10)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% RUN AFTER DLC:

analysis = 'GPe_A2a_ChR2_0p25mw'
behavior_str = '10x' # 'zone_'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,behavior_str=behavior_str)

# analysis = 'GPe_CAG_Arch'
# inc,exc,color = dataloc.experiment_selector(analysis,'zone_1')

ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = False)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,
                                             force_replace=True,win=10)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     

print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% Plot 10x10 / openloop days:
# ex0=['exclude','Bad','GPe','bad','Broken','15min', '10hz','grooming','Exclude','Other XLS']
# exc=[ex0]
# inc=[['AG','Str','A2A','Ai32','10x10','0p25mw']]


analysis = 'GPe_A2a_ChR2_2mw'
b = '10x' 
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,
                                                          behavior_str=b)
inc = inc[1] + ['AG7192']
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=False
for pn in pns:
    df,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_openloop_day(df,meta,save=saveit, close=closeit)

# %% Plot zone day
# inc=[['AG','Str','A2A','Ai32','zone_2','_0p5mw']]

analysis = 'GPe_A2a_ChR2_0p25mw'
behavior_str = 'zone_' #'10x'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,behavior_str=behavior_str)
inc[1] += ['AG7128']
pns=dataloc.raw_csv(basepath,inc[0],exc[0])

if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=False
for pn in pns:
    raw,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_zone_day(raw,meta,save=saveit, close=closeit)
    
# %% Plot zone day summary:   
# inc=[['AG','Str','A2A','Ai32','zone','_0p5mw']]

# analysis = 'Str_A2a_ChR2_1mw'
# analysis = 'Str_D1_Arch_30mw'
# analysis = 'GPe_CAG_Arch_1mw'
analysis = 'GPe_A2a_ChR2_0p25mw'
behavior_str = 'zone_' #'10x'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,'zone_1')
inc= [inc[1] + ['AG7192'], inc[1] + ['AG7128']]
# inc = [['AG','GPe','A2A','ChR2','zone_1','0p25mw']]
data = behavior.zone_rtpp_summary_collect(basepath,inc,exc,
                                          stim_analyze_dur=10,
                                          zone_analyze_dur= 10 * 60)


plots.plot_zone_mouse_summary(data,color='k',
                              example_mouse=example_mouse)



# %% Plot free-running / unstructured openfield data:
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=True
for pn in pns:
    df,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc')
    plots.plot_freerunning_day(df,meta,save=saveit, close=closeit)
    
# %% Load openloop mouse summary data:

# exc=[ex0]
# inc=[['AG','hm4di','Str','A2A','Ai32','saline','10x10','3mW']]
# inc=[['AG','Str','A2A','Ai32','10x10','0p25mw']]
# inc=[['AG','Str','A2A','Ai32','zone_1_0p5mw',]]
# inc=[['AG','Str','A2A','Ai32','zone','_0p5mw']]
# inc=[['AG','GPe','CAG','Arch','10x30',]]

# inc=[['AG','Str','A2A','ChR2','10x','Bilateral'],
#      ['AG','Str','A2A','Ai32','10x','Bilateral']]
# ex0=['exclude','Bad','GPe','bad',\
#      'Broken','15min','10hz', 'Exclude','Other XLS','AG3233_5','AG3233_4',\
#          'AG3488_7','_0p5mw','_0p25mw','gpe','muscimol','cno','hm4di','3mW','2mw']

# ex0=['exclude','Bad','Str','bad',\
#      'Broken','15min','10hz', 'Exclude','Other XLS','AG3233_5','AG3233_4',\
#          'AG3488_7','_0p5mw','_0p25mw','str','muscimol','cno','hm4di','3mW','2mw']
# inc=[['AG','GPe','CAG','Arch','10x',]]

# exc=[ex0,]

# analysis = 'GPe_A2a_ChR2_2mw'
analysis = 'Str_A2a_ChR2_1mw'
# analysis = 'Str_D1_Arch_30mw'
analysis = 'GPe_CAG_Arch_1mw'
behavior_str = '10x'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,behavior_str=behavior_str)

# inc= [inc[1] + ['AG7192'], inc[1] + ['AG7128']]
data = behavior.open_loop_summary_collect(basepath,inc,exc,update_rear=True)

# Plot openloop mouse summary with statistics
smooth_amnt= [33, 33]
# smooth_amnt=[33 * 3]
fig,stats,ax =plots.plot_openloop_mouse_summary(data,smooth_amnt=smooth_amnt,method=[10,1])
# %%
plt.savefig('/home/brian/Dropbox/Manuscripts/Isett_Gittis_2021/Figure 1/narrow_a2a_10x30_n8_sumamary.pdf')
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
