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
import math

# %% Plot one 10x30 experiment day
inc=[['AG','GPe','CAG','Arch','10x30']]
exc=[['exclude','_and_Str','Left','Right','Other XLS','Exclude']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
raw,meta=ethovision_tools.csv_load(pns[1])
raw=ethovision_tools.add_amb_to_raw(raw,meta)

#The magic:
plots.plot_openloop_day(raw,meta)

# %% Debug new metric: meander
#DLC Measure:
dir_smooth=behavior.smooth_direction(raw,meta,use_dlc=True)
dir_smooth_etho=behavior.smooth_direction(raw,meta)

diff_angle=signal.angle_vector_delta(dir_smooth[0:-1],dir_smooth[1:],thresh=20,
                            fs=meta['fs'][0])
meander = behavior.measure_meander(raw,meta,use_dlc=True)
dist = raw['vel'] * (1 / meta.fs[0])
plt.close('all')
fig = plt.figure(figsize=(5,10))
ax1=plt.subplot(3,1,1)
plt.plot(raw['time'][1:],signal.log_modulus(meander))
plt.subplot(3,1,2,sharex=ax1)
plt.plot(raw['time'],dist)
plt.subplot(3,1,3,sharex=ax1)
plt.plot(raw['time'][1:],diff_angle)

plt.figure(),plt.scatter(dist[1:],diff_angle,np.sqrt(np.power(meander,2))/100,
                         'k', alpha=0.1)

plt.xlabel('Distance (cm)')
plt.ylabel(r'$\Delta$ direction (deg)')

# %% Compare mouse direction etho vs. smoothed ethovision:
dir = raw['dir']
dir_smooth_dlc=behavior.smooth_direction(raw,meta,use_dlc=True)
dir_smooth_etho=behavior.smooth_direction(raw,meta)
plt.figure(),
ax0=plt.subplot(3,1,1)
plt.plot(raw['time'],dir,'k')
plt.plot(raw['time'],dir_smooth_etho,'--r')
plt.plot(raw['time'],dir_smooth_dlc,'--g')
plt.ylabel('Direction (deg)')

plt.subplot(3,1,2,sharex=ax0)
plt.plot(raw['time'],dir_smooth_etho-dir)
plt.ylabel('Smooth-NotSmooth')

plt.subplot(3,1,3,sharex=ax0)
plt.plot(raw['time'],raw['elon'])
plt.plot(raw['time'],raw['fine_move'])
plt.ylabel('Elongation (cm)')
# %% 
amb_bouts=behavior.bout_analyze(raw,meta,'ambulation',stim_dur=30,min_bout_dur_s=1)
im_bouts=behavior.bout_analyze(raw,meta,'im',stim_dur=30,min_bout_dur_s=1)
fm_bouts=behavior.bout_analyze(raw,meta,'fine_move',stim_dur=30,min_bout_dur_s=1)
plt.figure(),plt.imshow(amb_bouts['cont_y'].T,aspect='auto',interpolation='none')

# %% Check that immobility vs. ambulation vs. fine movement are detected correctly:
fig = plt.figure(figsize=(20,5))

y=raw['vel']
m=np.nanmean(y)
raw=ethovision_tools.add_amb_to_raw(raw,meta)
fm=(raw['im'] == False) & (raw['ambulation']==False)
wtf=(raw['im'] == True) & (raw['ambulation']== True)

plt.plot(raw['time'],y,'k')
plt.plot(raw['time'],raw['im']*m, 'b')
plt.plot(raw['time'],raw['ambulation']*m,'--r')
plt.plot(raw['time'],raw['fine_move']*m,'--c')
plt.plot(raw['time'][wtf],np.ones((sum(wtf),1))*m,'ro')
plt.plot(raw['time'].values[[0,-1]],meta['amb_vel_thresh'][0:2],'--g')

# %% Debug adding DLC data columns to Raw*.csv:
# 
ex0=['exclude','_and_GPe','Left','Right','Other XLS','Exclude']
inc=[['AG','GPe','CAG','Arch','10x']]
exc=[ex0]
# inc=[['AG','Str','A2A','Ai32','10x'],['AG','Str','A2A','ChR2','10x']]
# exc=[ex0,ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
dlc_path=dataloc.gen_paths_recurse(basepath,inc[0],exc[0],'dlc_analyze.h5')
raw_path=dataloc.raw_csv(basepath,inc[0],exc[0])

ethovision_tools.add_dlc_to_csv(basepath,inc,exc,save=True)
# dlc = behavior.load_and_clean_dlc_h5(dlc_path[0])
# raw,meta=ethovision_tools.csv_load(raw_path[0])