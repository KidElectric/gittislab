#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:03:08 2020
Test creating plot analysis of 10x10 / 10x30 type data
@author: brian
"""

# %%
from gittislab import signal, behavior, dataloc, ethovision_tools, plots
import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem, t
import pdb
# inc=['GPi','CAG','Arch','10x30','AG6151_3_CS090720']
# inc=['AG','GPe','FoxP2','ChR2','10x10_20mW',]
# exc=['exclude','_and_Str','Left','Right']
# basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
# pns=dataloc.rawh5(basepath,inc,exc)
# raw_df,raw_par=ethovision_tools.h5_load(pns[0])
# %%
inc=[['AG','GPe','CAG','Arch','10x30']]
exc=[['exclude','_and_Str','Left','Right','Other XLS','Exclude']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
raw,meta=ethovision_tools.csv_load(pns[1])
clip=behavior.stim_clip_grab(raw,meta,y_col='vel')
clip_ave=behavior.stim_clip_average(clip)

# %% Create figure layout using gridspec:
fig = plt.figure(constrained_layout = True,figsize=(8,20))
gs = fig.add_gridspec(5, 3)

f_row=list(range(5))
f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
f_row[1]=fig.add_subplot(gs[1,:])
f_row[2]=[fig.add_subplot(gs[2,0:2]), 
        fig.add_subplot(gs[2,2])]
f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]

ax_pos = plots.trial_part_position(raw,meta,ax=f_row[0])
ax_speedbar = plots.mean_cont_plus_conf(clip_ave,xlim=[-10,20],highlight=[0,10,20],ax=f_row[2][0])
ax_speed = plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],ax=f_row[2][1])
