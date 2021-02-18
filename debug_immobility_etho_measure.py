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
from scipy import stats
# %% Debugging code incorporated into preprocessing:
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','mW','mw']

inc=[['AG','A2A','Ai32','10x10']]
exc=[ex0,ex0,ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.raw_csv_to_preprocessed_csv(basepath,
                                             inc,exc,force_replace=True,
                                             win=10)
# %% Plot comparing im & im2 (improved)

inc=['AG','Str','A2A','Ai32','10x10',]
exc=['exclude','gpe_','mw']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc,exc)
use=1
# fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG6343_5_BI120220/Raw_AG6343_5_BI120220.csv'
raw,meta=ethovision_tools.csv_load(pns[use],method='preproc')

plt.figure()
plt.plot(raw['time'],raw['im'])
plt.plot(raw['time'],raw['im2'])

# %%  Compare ethovision immobility measurement to using DLC side camera stats
# and or velocity threshold.

#Load an example file
inc=['AG','Str','A2A','Ai32','10x10',]
exc=['exclude','gpe_','mw']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc,exc)
use=1
# fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG6343_5_BI120220/Raw_AG6343_5_BI120220.csv'
raw,meta=ethovision_tools.csv_load(pns[use],method='preproc')
raw,meta = ethovision_tools.add_dlc_helper(raw,meta,pns[use].parent,inc,exc,force_replace=True)
# plots.plot_openloop_day(raw,meta);


im = raw['im'].values.astype(float)
vel=raw['vel'].values
vel[0:5]=np.nan
feats=['dlc_snout_x','dlc_snout_y','dlc_side_left_hind_x', 'dlc_side_left_hind_y','dlc_side_right_fore_x',
       'dlc_side_right_fore_y', 'dlc_side_tail_base_x', 'dlc_side_tail_base_y',
       'dlc_side_left_fore_x','dlc_side_right_hind_x', 'dlc_side_right_hind_y',
       'dlc_side_left_fore_y','dlc_head_centroid_x', 'dlc_head_centroid_y',]
d=np.ones(im.shape) * 0
step=20


x=raw['dlc_top_body_center_y'].values
        

        
for c in feats:
    dat = raw[c].values
    dat=signal.max_correct(x,dat,step,poly_order=2) #Correct for distance from camera
    dd = abs(np.diff(np.hstack((dat[0],dat))))
    d=np.vstack((d,dd))
d=np.nanmax(d,axis=0)


vel=raw['vel']
# head_crit=0.6 #For raw pix
head_crit = 0.004 # For position normalized
vel_crit = 0.5
plt.figure(),
ax0=plt.subplot(3,1,1)
plt.plot(raw['time'],im,'k')
# plt.plot(raw['time'],raw['rear'],'b')
plt.ylabel('Occurence')

plt.subplot(3,1,2,sharex=ax0)
plt.plot(raw['time'],d,'k')
plt.plot(raw['time'],raw['mouse_height'].values,'r')
plt.plot([0,meta['exp_end'][0]],[head_crit,head_crit],'--b')
# plt.ylim([0,0.01])
plt.ylabel('Norm. DLC change (average)')

plt.subplot(3,1,3,sharex=ax0)
# plt.plot(raw['time'],raw['mouse_height'],'k')
plt.plot(raw['time'],vel,'r')
plt.ylim([0,1])
plt.plot([0,meta['exp_end'][0]],[vel_crit,vel_crit],'--b')
plt.ylabel('Height and vel')

plt.sca(ax0)
height=raw['mouse_height']
# new_crit = np.array((dhx < head_crit ) & (dhy < head_crit) & (vel < vel_crit))
new_crit = np.array((d < head_crit) & (vel < vel_crit) & (height < 0.2))
smooth_crit = 0.2
new_crit_temp = signal.boxcar_smooth(new_crit,round(meta['fs'][0]*0.5)) 
new_crit =  new_crit_temp >= smooth_crit
plt.plot(raw['time'],new_crit_temp,'--r')
plt.plot([0,meta['exp_end'][0]],[smooth_crit,smooth_crit],'--b')
plt.plot(raw['time'],new_crit,'--g')
r,p= stats.pearsonr(new_crit,im)
plt.title('Pearson r = %1.3f, p = %1.4f' % (r,p) )
# %% Examine in boris:
raw,meta=ethovision_tools.csv_load(pns[use],method='preproc')
raw['im2'] = new_crit
preproc_file_name=dataloc.path_to_preprocfn(pns[use]) + '.csv'
raw.to_csv(pns[use].parent.joinpath(preproc_file_name))

basepath=pns[use].parent
inc=[['AG']]
exc=[['exclude']]
boris,raw,meta=ethovision_tools.boris_prep(basepath,inc,exc,plot_cols=['time','im','im2'], 
                            event_col=['im','im2'],event_thresh=0.5, method='preproc')
