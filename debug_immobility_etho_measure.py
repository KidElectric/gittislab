#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:23:59 2021

@author: brian
"""

from gittislab import signal, behavior, dataloc, ethovision_tools, plots 
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import pdb
from itertools import compress
from matplotlib import pyplot as plt

# %%  Compare ethovision immobility measurement to using DLC side camera stats
# and or velocity threshold.

#Load an example file
inc=['AG','Str','A2A','Ai32','10x10',]
exc=['exclude','gpe_','mw']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc,exc)
# fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG6343_5_BI120220/Raw_AG6343_5_BI120220.csv'
raw,meta=ethovision_tools.csv_load(pns[0],method='preproc')
raw,meta = ethovision_tools.add_dlc_helper(raw,meta,pns[0].parent,inc,exc,force_replace=True)
plots.plot_openloop_day(raw,meta)

# %% 
im = raw['im']
vel=raw['vel'].values
vel[0:5]=np.nan
feats=['dlc_side_left_hind_x', 'dlc_side_left_hind_y','dlc_side_right_fore_x',
       'dlc_side_right_fore_y', 'dlc_side_tail_base_x', 'dlc_side_tail_base_y',
       'dlc_side_left_fore_x','dlc_side_right_hind_x', 'dlc_side_right_hind_y',
       'dlc_side_left_fore_y','dlc_head_centroid_x', 'dlc_head_centroid_y',]
d=np.ones(im.shape) * 0
for c in feats:
    dat = raw[c].values
    dd = abs(np.diff(np.hstack((dat[0],dat))))
    d=np.vstack((d,dd))
d=np.nanmean(d,axis=0)
# head_x=raw['dlc_head_centroid_x'].values
# dhx= abs(np.diff(np.hstack((head_x[0],head_x))))
# head_y=raw['dlc_head_centroid_y'].values
# dhy = abs(np.diff(np.hstack((head_y[0],head_y))))
# head_x=raw['dlc_head_centroid_x'].values
# dhx= abs(np.diff(np.hstack((head_x[0],head_x))))
# head_y=raw['dlc_head_centroid_y'].values
# dhy = abs(np.diff(np.hstack((head_y[0],head_y))))

vel=raw['vel']
head_crit=0.6
vel_crit = 0.5
plt.figure(),
ax0=plt.subplot(3,1,1)
plt.plot(raw['time'],im,'k')
# plt.plot(raw['time'],raw['rear'],'b')
plt.ylabel('Occurence')

plt.subplot(3,1,2,sharex=ax0)
# plt.plot(raw['time'],head_x,'k')
plt.plot(raw['time'],d,'k')

# plt.plot(raw['time'],head_y,'r')
# plt.plot(raw['time'],dhy,'--r')

plt.plot([0,meta['exp_end'][0]],[head_crit,head_crit],'--b')
plt.ylim([0,1])
plt.ylabel('Head x-y')

plt.subplot(3,1,3,sharex=ax0)
# plt.plot(raw['time'],raw['mouse_height'],'k')
plt.plot(raw['time'],vel,'r')
plt.ylim([0,1])
plt.plot([0,meta['exp_end'][0]],[vel_crit,vel_crit],'--b')
plt.ylabel('Height and vel')

plt.sca(ax0)
# new_crit = np.array((dhx < head_crit ) & (dhy < head_crit) & (vel < vel_crit))
new_crit = np.array((d < head_crit) & (vel < vel_crit))
new_crit = signal.boxcar_smooth(new_crit,round(meta['fs'][0])*1)  > 0.25
plt.plot(raw['time'],new_crit,'--r')

# %% Examine in boris:
raw,meta=ethovision_tools.csv_load(pns[0],method='preproc')
raw['im2'] = new_crit
preproc_file_name=dataloc.path_to_preprocfn(pns[0]) + '.csv'
raw.to_csv(pns[0].parent.joinpath(preproc_file_name))

basepath=pns[0].parent
inc=[['AG']]
exc=[['exclude']]
boris,raw,meta=ethovision_tools.boris_prep(basepath,inc,exc,plot_cols=['time','im','im2'], 
                            event_col=['im','im2'],event_thresh=0.5, method='preproc')
