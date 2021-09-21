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
# %% ID Crossings:
    
    
ac_on,ac_off= signals.thresh(raw['iz1'].astype(int),0.5,'Pos')
min=0 #meta['fs'][0] * 4 #4 seconds
all_cross=[]
cross_t=[]
fs= meta['fs'][0]
for on,off in zip(ac_on,ac_off):
    if (off-on) > min:
        all_cross.append([on,off])
        cross_t.append(on/fs)
durs = np.diff(np.array(all_cross),axis=1) / fs
print('%d crossings detected. Median dur: %1.2fs' % \
      (len(all_cross),np.median(durs)))

maxt=round(meta.task_start[0]/60)+10 #should be 20 minutes
t0=0
binsize=1 #minutes
all_on=(np.array(all_cross)[:,0]/fs) / 60 #in minutes
time_bins = np.array([x for x in range (t0,maxt+binsize,binsize)])
binned_counts=[]
med_dur=[]
keep_t=[]

for t0,t1 in zip(time_bins[0:-1],time_bins[1:]):
    ind = (all_on > t0) & (all_on <=t1)
    binned_counts.append(np.sum(ind))
    if any(ind):
        med_dur.append(np.median(durs[ind]))
    else:
        med_dur.append(np.nan)
    
    keep_t.append(t0+((t1-t0)/2))


t0=round(meta.task_stop[0]/60)
maxt=t0 + 10
time_bins = np.array([x for x in range (t0,maxt+binsize,binsize)])

for t0,t1 in zip(time_bins[0:-1],time_bins[1:]):
    ind = (all_on > t0) & (all_on <=t1)
    binned_counts.append(np.sum(ind))
    if any(ind):
        med_dur.append(np.median(durs[ind]))
    else:
        med_dur.append(np.nan)
    keep_t.append(t0+((t1-t0)/2))

# plt.figure()
# plt.plot(keep_t,med_dur,'o-')
# plt.plot(keep_t,binned_counts,'o-')
        #     dist_mat[i,:],_ = np.histogram(-dist,dist_bins)
        #     ii += 1
        #     tot_hist=np.nansum(dist_mat,axis=0)
        #     tot_hist = 100*(tot_hist/np.nansum(tot_hist))
        # a.bar(dist_bins[0:-1] - width/2,tot_hist,width,)    
        # a.set_ylabel('% of 8s Crossing')
        # a.set_xlabel('Dist (cm)')
        # a.set_ylim([0,10])
# new_meta=meta
# baseline=4 # seconds before / after crossing
# for i,cross in enumerate(all_cross):
#     new_meta.loc[i,'stim_on']=raw['time'][cross[0]]
#     new_meta.loc[i,'stim_off']=raw['time'][cross[0]]
# fs=meta['fs'][0]
# on_time =raw['time'][np.array(all_cross)[:,0]].values
    
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

ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','zone_1_d1r',
     '_gpe_muscimol','_gpe_pbs','mW','mw']

inc=[['AG','GPe','CAG','Arch','zone_1'],
     ['AG','GPe','A2A','Ai32','zone_1'],
     ['AG','Str','A2A','Ai32','zone_1'],
     ['AG','Str','A2A','ChR2','zone_1'],
     ]

# inc = [inc[0]]

exc=[ex0,ex0,ex0,ex0]
lens=[]
all_dur=[]
all_count=[]
all_time=[]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
for i,e in zip (inc,exc):
    pns=dataloc.raw_csv(basepath,i,e)
    
    for pn in pns:
        raw,meta=ethovision_tools.csv_load(pn,method='preproc' )
        # data=behavior.experiment_summary_helper(raw, meta)
        plots.plot_zone_day(raw,meta)
        # # plots.zone_day_crossing_stats(raw,meta)
        # a,b,c=behavior.measure_crossings(raw,meta)
        # all_count.append(a)
        # all_dur.append(b)
        # all_time.append(c)
        # print(pn)
        # print(len(c))
        # lens.append(len(c))
    
#Row 4: approaches that do not lead to crossing.

#Row 5: Micro-crossing, %crossing in center vs. edges, other parameters?

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