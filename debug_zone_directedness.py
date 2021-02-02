#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:17:56 2021

@author: brian
"""

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
from scipy.signal import find_peaks 
from scipy.stats import circmean
from statistics import mode


# %%
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','zone_1_d1r',
     '_gpe_muscimol','_gpe_pbs','mW','mw']

inc=[['AG','GPe','CAG','Arch','zone_1'],
     ['AG','Str','A2A','Ai32','zone_1'],
     ['AG','Str','A2A','ChR2','zone_1']]
exc=[ex0,ex0,ex0]

basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
# ethovision_tools.add_dlc_to_csv(basepath,inc,exc,save=True)
pns=dataloc.raw_csv(basepath,inc[1],ex0)
raw,meta=ethovision_tools.csv_load(pns[1],method='preproc' )
r,_=ethovision_tools.csv_load(pns[1])
# raw,meta=ethovision_tools.csv_load(pns[0])

# %%
plots.plot_zone_day(raw,meta);

# %%
x_s=raw['x']
y_s=raw['y']
cross=np.concatenate(([0],np.diff(raw['iz1'].astype(int)) > 0)).astype('bool')
cross_zero=np.median(x_s[cross])
print(cross_zero)
plt.figure()
plt.plot(x_s,y_s)
plt.plot(x_s-cross_zero,y_s)

# %% Plot approaches to zone 1:
c,nc=behavior.z2_to_z1_cross_detect(raw,meta)
ac_on,ac_off= signal.thresh(raw['iz1'].astype(int),0.5,'Pos')
min=meta['fs'][0] #1 second
all_cross=[]
for on,off in zip(ac_on,ac_off):
    if (off-on) > min:
        all_cross.append([on,off])
a=np.zeros((len(all_cross),1))

    
plt.figure()
plt.plot([-25,-25,25,25,-25],[-25,25,25,-25,-25],'k')
plt.plot([0,0],[-25,25],'--k')
for i,j in c:
    fullx=raw['x'][i:j]
    plt.plot(fullx,raw['y'][i:j],10,c='r')

for i,j in nc:
    fullx=raw['x'][i:j]
    plt.plot(fullx,raw['y'][i:j],10,c='c')

for cross in all_cross:
    x=raw['x'][cross[0]]
    y=raw['y'][cross[0]]
    plt.plot(x,y,'ob')
    
# %% For all crossings > 4 seconds, grab velocity trace around crossing +/- 4 seconds
#Sort these for each period: baseline, during stim, post stim
ac_on,ac_off= signal.thresh(raw['iz1'].astype(int),0.5,'Pos')
min=meta['fs'][0] * 4 #4 seconds
all_cross=[]
for on,off in zip(ac_on,ac_off):
    if (off-on) > min:
        all_cross.append([on,off])
print('%d crossings detected.' % len(all_cross))
new_meta=meta
baseline=4 # seconds before / after crossing
for i,cross in enumerate(all_cross):
    new_meta.loc[i,'stim_on']=raw['time'][cross[0]]
    new_meta.loc[i,'stim_off']=raw['time'][cross[0]]
vel_cross=behavior.stim_clip_grab(raw,new_meta,'vel',x_col='time',
                        stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)

t=[i for i in range(4)]
t[0]=0
t[1]=meta.task_start[0]
t[2]=meta.task_stop[0]
t[3]=meta.exp_end[0]

fig = plt.figure(constrained_layout = True,figsize=(9.48, 7.98))
gs = fig.add_gridspec(4, 3)
f_row=list(range(gs.nrows))
f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
f_row[1]=[fig.add_subplot(gs[1,i]) for i in range(3)]
f_row[2]=[fig.add_subplot(gs[2,i]) for i in range(3)]
f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]

# f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
# f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]

#Row 0 vel clips full colorscale:
on_time =raw['time'][np.array(all_cross)[:,0]].values
fs=meta['fs'][0]
xtick_samps=[x for x in range(0,vel_cross['cont_y'].shape[0]+1,round(fs*2))]
xticklab=[str(round(x)) for x in (xtick_samps/fs)-baseline]
for i,a in enumerate(f_row[0]):
    ind = (on_time > t[i]) & (on_time < t[i+1])
    a.imshow(vel_cross['cont_y'][:,ind].T,
             aspect='auto')
    plt.sca(a)
    plt.xticks(xtick_samps,xticklab)
    yticks=[x for x in range (0,sum(ind),5)]
    yticklab=[str(y) for y in yticks]
    plt.yticks(yticks,yticklab)
    a.set_xlabel('Time (s)')
    a.set_ylabel('Cross #')

    
#Row 1: Plot beginning, middle, and late crossing mean velocity:
labels=['Early','Middle','Late']
for i,a in enumerate(f_row[1]):
    ind = (on_time > t[i]) & (on_time < t[i+1])
    cross_period=vel_cross['cont_y'][:,ind]
    ncross=sum(ind)
    step=round(ncross/3)
    eml =[x for x in range(0,ncross,step)]
    eml = eml[0:3]
    for ii,j in enumerate(eml) :
        vel_stage=cross_period[:,j:(j+step)]
        x=vel_cross['cont_x']
        y=np.mean(vel_stage,axis=1)
        a.plot(x,y,label=labels[ii])
        a.set_ylim([0,30])
    if i == 2:    
        a.legend()
        a.set_xlabel('Time (s)')
        a.set_ylabel('Speed (cm/s)')
    
# %% Summarize across many mice / conditions:
data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'room','cross_per','cross_counts'])
temp=data
min_bout=1
use_dlc=False
keep_enter=[]
keep_exit=[]
use_col=['iz1','iz2','dir','vel','x','ambulation']
for ii,ee in zip(inc,exc):
    pns=dataloc.raw_csv(basepath,ii,ee)
    for pn in pns:
        temp={}
        raw,meta=ethovision_tools.csv_load(pn,columns=use_col,method='preproc')
        temp['anid']=meta['anid'][0]
        temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                                 meta['stim_area'][0],
                                                 meta['opsin_type'][0])
        temp['proto']=meta['protocol'][0]
        temp['room']=meta.exp_room_number[0]
        c,nc=behavior.z2_to_z1_cross_detect(raw,meta,
                                            start_cross_dist=10,
                                            stop_cross_dist=10,
                                            max_cross_dur=5,
                                            min_total_dist=5,
                                            min_cross_dist=3)
        tot_c= behavior.trial_part_count_cross(c,nc,meta)
        completed_cross=(tot_c[0,:]/np.nansum(tot_c,axis=0)) * 100
        temp['cross_per']= completed_cross
        temp['cross_counts']=np.nansum(tot_c,axis=0)
        
        c,nc=behavior.z2_to_z1_cross_detect(raw,meta,
                                            start_cross_dist=15,
                                            stop_cross_dist=10, #Not enforced, just defines end of crossing
                                            max_cross_dur=10, #Enforced
                                            min_total_dist=10, #Enforced
                                            min_cross_dist=3)
        
        tot_c= behavior.trial_part_count_cross(c,nc,meta)
        completed_cross=(tot_c[0,:]/np.nansum(tot_c,axis=0)) * 100
        temp['far_cross_per']= completed_cross
        
        on,off=signal.thresh(raw['iz1'].astype(int),0.5,'Pos')
        facing=circmean(raw['dir'][on],high=180,low=-180)
        temp['face_z1_dir']=facing
        data=data.append(temp,ignore_index=True)
# %% Plot this
use_field='far_cross_per'
plt.figure()
ax=plt.subplot(1,2,1)
plt.title('Str A2a ChR2')
chr2= np.array([('Ai32'in x or 'ChR2' in x) for x in data['cell_area_opsin']])
subset=np.stack(list(data.loc[chr2,use_field]),axis=0)
clip={'data':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='data', ax=ax,
                         clip_method=False)
plt.ylabel('% Of crossings completed')
plt.ylim(0,100)


ax=plt.subplot(1,2,2)
plt.title('GPe CAG Arch')
subset=np.stack(list(data.loc[~chr2,use_field]),axis=0)
clip={'data':subset}
plots.mean_bar_plus_conf(clip, ['Pre','Dur','Post'],
                         use_key='data',ax=ax,
                         clip_method=False)
plt.ylim(0,100)
plt.ylabel('% Of crossings completed')
# %% Where are rears happening during zone task?