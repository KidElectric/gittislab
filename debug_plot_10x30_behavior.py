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
raw=ethovision_tools.add_amb_to_raw(raw,meta)
baseline= round(np.mean(meta['stim_dur']))
stim_dur= baseline
# Calculate stim-triggered speed changes:
vel_clip=behavior.stim_clip_grab(raw,meta,y_col='vel', stim_dur=stim_dur)

clip_ave=behavior.stim_clip_average(vel_clip)

# Calculate stim-triggered %time mobile:
percentage = lambda x: (np.nansum(x)/len(x))*100
m_clip=behavior.stim_clip_grab(raw,meta,y_col='m', stim_dur=stim_dur, summarization_fun=percentage)

#Calculate

# %% Create figure layout using gridspec:
plt.close('all')
fig = plt.figure(constrained_layout = True,figsize=(8.5,11))
gs = fig.add_gridspec(6, 3)
stim_dur=np.mean(meta['stim_dur'])
f_row=list(range(gs.nrows))
f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
f_row[1]=[fig.add_subplot(gs[1,0:2]) , fig.add_subplot(gs[1,2])]
f_row[2]=[fig.add_subplot(gs[2,0:2]) , fig.add_subplot(gs[2,2])]
f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]

ax_pos = plots.trial_part_position(raw,meta,ax=f_row[0])
plt.sca(ax_pos[1])
plt.title('%s, %s   %s   %s %s %s' % (meta['anid'][0],
                      meta['etho_exp_date'][0],
                      meta['protocol'][0],
                      meta['cell_type'][0],
                      meta['opsin_type'][0],
                      meta['stim_area'][0]))
ax_speedbar = plots.mean_cont_plus_conf(clip_ave,xlim=[-stim_dur,stim_dur*2],
                                        highlight=[0,stim_dur,25],ax=f_row[1][0])
plt.ylabel('Speed (cm/s)')
plt.xlabel('Time from stim (s)')

ax_speed = plots.mean_bar_plus_conf(vel_clip,['Pre','Dur','Post'],ax=f_row[1][1])
plt.ylabel('Speed (cm/s)')
plt.xlabel('Time from stim (s)')

ax_im = plots.mean_bar_plus_conf(m_clip,['Pre','Dur','Post'],ax=f_row[2][0])
plt.ylabel('% Time Mobile')

amb_bouts=behavior.bout_analyze(raw,meta,'ambulation',stim_dur=30,min_bout_dur_s=0.5)
im_bouts=behavior.bout_analyze(raw,meta,'im',stim_dur=30,min_bout_dur_s=0.5)


######################
#Ambulation bout row
#Rate
ax_amb_bout_rate= plots.mean_bar_plus_conf(amb_bouts,['Pre','Dur','Post'],
                                           use_key='rate',ax=f_row[3][0])
plt.ylabel('Amb. bouts / 30s')

#Duration
ax_amb_bout_dur = plots.mean_bar_plus_conf(amb_bouts,['Pre','Dur','Post'],
                                           use_key='dur',ax=f_row[3][1])
plt.ylabel('Amb. dur (s)')

#Speed
ax_amb_bout_speed= plots.mean_bar_plus_conf(amb_bouts,['Pre','Dur','Post'],
                                           use_key='speed',ax=f_row[3][2])
plt.ylabel('Amb. speed (cm/s)')

##################################
#Immobility bout row
#Rate
ax_im_bout_rate= plots.mean_bar_plus_conf(im_bouts,['Pre','Dur','Post'],
                                           use_key='rate',ax=f_row[4][0])
plt.ylabel('Im. bouts / 30s')

#Duration
ax_im_bout_dur= plots.mean_bar_plus_conf(im_bouts,['Pre','Dur','Post'],
                                           use_key='dur',ax=f_row[4][1])
plt.ylabel('Im. dur (s)')

ax_im_bout_speed= plots.mean_bar_plus_conf(im_bouts,['Pre','Dur','Post'],
                                           use_key='speed',ax=f_row[4][2])
plt.ylabel('Im. speed (cm/s)')

###########################
#Meander row

# %% Debug new metric: meander
dir_smooth=behavior.smooth_direction(raw,meta)
diff_angle=signal.angle_vector_delta(dir_smooth[0:-1],dir_smooth[1:],thresh=20,
                            fs=meta['fs'][0])
meander = behavior.measure_meander(raw,meta)
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
