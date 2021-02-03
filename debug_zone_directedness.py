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
fs=meta['fs'][0]
on_time =raw['time'][np.array(all_cross)[:,0]].values
vel_cross=behavior.stim_clip_grab(raw,new_meta,'vel',x_col='time',
                        stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)
x_cross=behavior.stim_clip_grab(raw,new_meta,'x',x_col='time',
                        stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)
y_cross=behavior.stim_clip_grab(raw,new_meta,'y',x_col='time',
                        stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)

t=[i for i in range(4)]
t[0]=0
t[1]=meta.task_start[0]
t[2]=meta.task_stop[0]
t[3]=meta.exp_end[0]

fig = plt.figure(constrained_layout = True,figsize=(9.48, 7.98))
col_labs=['Pre','Dur','Post']
gs = fig.add_gridspec(5, 3)
f_row=list(range(gs.nrows))
f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
f_row[1]=[fig.add_subplot(gs[1,i]) for i in range(3)]
f_row[2]=[fig.add_subplot(gs[2,i]) for i in range(3)]
f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
# f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]


# Row 0: Plot trajectories overlayed
r=0
xtick_samps=np.array([x for x in range(0,x_cross['cont_y'].shape[0]+1,round(fs*2))])
xticklab=[str(round(x)) for x in (xtick_samps/fs)-baseline]
zero_samp=int(xtick_samps[-1]/2)
for i,a in enumerate(f_row[r]):
    ind = (on_time > t[i]) & (on_time < t[i+1])
    cross_x=x_cross['cont_y'][:,ind]
    cross_y=y_cross['cont_y'][:,ind]
    for x,y in zip(cross_x.T,cross_y.T):
        b=y[zero_samp]
        y= y-b
        a.scatter(x,y,2,alpha=0.1,facecolor='k',)
        
    a.plot([0,0],[-25,25],'--r')
    a.set_xlabel('Dist (cm)')
    a.set_title(col_labs[i])
    
    # ncross=sum(ind)
    # step=round(ncross/3)
    # eml =[x for x in range(0,ncross,step)]
    # eml = eml[0:3]
    # for ii,j in enumerate(eml):
    #     vel_stage=cross_period[:,j:(j+step)]
    #     x=vel_cross['cont_x']

#Row 1 vel clips full colorscale:
r=1
xtick_samps=[x for x in range(0,vel_cross['cont_y'].shape[0]+1,round(fs*2))]
xticklab=[str(round(x)) for x in (xtick_samps/fs)-baseline]

for i,a in enumerate(f_row[r]):
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
    
    
#Row 2: Plot beginning, middle, and late crossing mean velocity:
labels=['Early','Middle','Late']
keep=np.ones((xtick_samps[-1],3))
r=2
for i,a in enumerate(f_row[r]):
    ind = (on_time > t[i]) & (on_time < t[i+1])
    cross_period=vel_cross['cont_y'][:,ind]
    ncross=sum(ind)
    step=round(ncross/3)
    eml =[x for x in range(0,ncross,step)]
    eml = eml[0:3]
    for ii,j in enumerate(eml):
        vel_stage=cross_period[:,j:(j+step)]
        x=vel_cross['cont_x']
        y=np.mean(vel_stage,axis=1)
        # if i > 0:
        #      y = y - baseline_change
        if i==0:
            keep[:,ii]=y
        a.plot(x,y,label=labels[ii])
        a.set_ylim([0,30])
    if i ==0:
        baseline_change=np.mean(keep,axis=1)
        # a.plot(x,np.mean(keep,axis=1),'--k')
    if i == 2:    
        a.legend()
        
    a.set_xlabel('Time (s)')
    a.set_ylabel('Speed (cm/s)')

# Row 3: Speed vs. distance from crossing
r=3
# Bin velocity of cross by x-distance of cross:
dist_bins=[x for x in range(-16,16+1,2)]
ncross_tot=vel_cross['cont_y'].shape[1]
dist_mat=np.ones((ncross_tot,len(dist_bins)))
i=0
for dist,vel in zip(x_cross['cont_y'].T,vel_cross['cont_y'].T):
    ind = np.digitize(-dist,dist_bins)
    for x in range(np.min(ind),np.max(ind)):
        use= ind == x
        if any(use):
            dist_mat[i,x]=np.mean(vel[use])
        else:
            dist_mat[i,x]=np.nan
    i += 1
xtick_samps=np.array([x for x in range(0,len(dist_bins),1)])
xticklab=[str(round(x)) for x in dist_bins]

for i,a in enumerate(f_row[r]):
    ind = (on_time > t[i]) & (on_time < t[i+1])
    a.imshow(dist_mat[ind,:], aspect='auto')
    plt.sca(a)
    plt.xticks(xtick_samps,xticklab)
    yticks=[x for x in range (0,sum(ind),5)]
    yticklab=[str(y) for y in yticks]
    plt.yticks(yticks,yticklab)
    a.set_xlabel('Dist from SZ (cm)')
    a.set_ylabel('Cross #')

#Row 4: Plot beginning, middle, and late crossing mean velocity:
labels=['Early','Middle','Late']
r=4
keep=np.ones((len(xtick_samps),3))
for i,a in enumerate(f_row[r]):
    ind = (on_time > t[i]) & (on_time < t[i+1])
    cross_period=dist_mat[ind,:]
    ncross=sum(ind)
    step=round(ncross/3)
    eml =[x for x in range(0,ncross,step)]
    eml = eml[0:3]
    for ii,j in enumerate(eml):
        vel_stage=cross_period[j:(j+step),:]
        x=dist_bins
        y=np.nanmean(vel_stage,axis=0)
        if i==0:
            keep[:,ii]=y
        a.plot(x,y,label=labels[ii])
        a.set_ylim([0,30])
    # if i ==1:
        # a.plot(x,np.mean(keep,axis=1),'--k')
    if i == 2:    
        a.legend()
        
    a.set_xlabel('Dist from SZ (cm)')
    a.set_ylabel('Speed (cm/s)')


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