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

# %% Overlay speed summaries saline vs. cno:
# plt.figure()
ax_speedbar=None
ax=[]
cols=['b','r']
for cond in conds:
        data=keep[cond]
        dat=data.loc[:,'vel_bin'].values
        x=data.loc[:,'x_bin'].values
        y=np.vstack([x for x in dat])
        ym= np.mean(y,axis=0)
        clip_ave={'cont_y' : ym,
                  'cont_x' : x[0]/60,
                  'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
                  'disc' : np.vstack(data['amb_speed'].values)}
        dur = 15
        ax_speedbar = plots.mean_cont_plus_conf(clip_ave,
                                          highlight=None,
                                          xlim=[0,dur],
                                          ax=ax_speedbar)
lines=ax_speedbar.get_lines()
for cond,col,line in zip(conds,cols,lines):
    line.set_label(cond)
    line.set_color(col)
plt.legend()

# %% Compare %Time mobile
sns.set_theme(style="whitegrid")
df = pd.DataFrame()
for cond in conds:
    temp=keep[cond].loc[:,('anid','per_mobile')]
    temp=temp.sort_values(by=['anid'])
    label_columns=['0-5','5-10','10-15']
    temp2=pd.DataFrame(temp['per_mobile'].to_list(),columns=label_columns)
    temp2['anid']=temp['anid']
    temp2['cond']=cond
    df= pd.concat((df,
        table_wrappers.consolidate_columns_to_labels(temp2,
                                                     label_columns,
                                                     value_column_name='per_mobile',
                                                     label_column_name='time_window')
        ))



ax = sns.barplot(x="time_window", y="per_mobile",hue='cond', data=df)
ax.set_xlabel('Minutes')
ax.set_ylabel('Time mobile (%)')