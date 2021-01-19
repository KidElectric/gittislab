#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 21:11:14 2021

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
     'Other XLS','Exclude','mW','mw']

inc=[['AG','10x10','_gpe_muscimol'],
     ['AG','10x10','_gpe_pbs'],
     ['AG','zone_','AG6343_4'],
     ['AG','zone_','AG6343_5'],]
exc=[ex0,ex0,ex0,ex0]

basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'room','speed','mobile','amb_speed','amb_count',
                              'in_sz'])
temp=data
min_bout=1
use_dlc=False
keep_enter=[]
percentage = lambda x: round((np.nansum(x)/len(x))*100,ndigits=2)
for ii,ee in zip(inc,exc):
    pns=dataloc.raw_csv(basepath,ii,ee)
    for pn in pns:
        temp={}
        raw,meta=ethovision_tools.csv_load(pn)
        raw=ethovision_tools.add_amb_to_raw(raw,meta)
        temp['anid']=meta['anid'][0]
        temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                                 meta['stim_area'][0],
                                                 meta['opsin_type'][0])
        temp['proto']=meta['protocol'][0]
        temp['room']=meta.exp_room_number[0]
        
        
        if 'zone' in temp['proto']:
            if meta['zone'][0] == 'Zone 1':
                use = 'iz1'
            else:
                use = 'iz2'
                
            t=[i for i in range(4)]
            t[0]=0
            t[1]=meta.task_start[0]
            t[2]=meta.task_stop[0]
            t[3]=meta.exp_end[0]
            in_zone=[]
            for i in range(len(t)-1):
                ind=(raw['time']>= t[i]) & (raw['time'] < t[i+1])
                in_zone.append(percentage(np.array(raw[use].astype(int))[ind]))
            temp['in_sz']=in_zone
 
        else:
            stim_dur = round(np.mean(meta['stim_dur']))        
            vel_clip=behavior.stim_clip_grab(raw,meta,
                                              y_col='vel', 
                                              stim_dur=stim_dur)        
            clip_ave=behavior.stim_clip_average(vel_clip)   
            temp['speed']=clip_ave['disc_m'].flatten()
            
            #### Calculate stim-triggered %time mobile:

            m_clip=behavior.stim_clip_grab(raw,meta,y_col='m', 
                                            stim_dur=stim_dur, 
                                            summarization_fun=percentage)        
            temp['mobile']=np.mean(m_clip['disc'],axis=0)
            
            #### Calculate ambulation bout properties:
            amb_counts=behavior.bout_analyze(raw,meta,'ambulation',
                                            stim_dur=stim_dur,
                                            min_bout_dur_s=min_bout,
                                            use_dlc=use_dlc)
            temp['amb_speed']=np.nanmean(amb_counts['speed'],axis=0)
            temp['amb_count']=np.nanmean(amb_counts['count'],axis=0)
        data=data.append(temp,ignore_index=True)
# %% Plot normalized % time in zone:
plt.figure()
ax=plt.subplot(1,2,2)
plt.title('Muscimol in GPe')
zone=np.array([(('zone' in x) and ('muscimol' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[zone,'in_sz']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'in_sz':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='in_sz',ax=ax,
                         clip_method=False)
plt.ylabel('% time in SZ')


ax=plt.subplot(1,2,1,sharey=ax)
plt.title('No muscimol in GPe')
zone=np.array([(('zone' in x) and not ('muscimol' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[zone,'in_sz']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'in_sz':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='in_sz',ax=ax,
                         clip_method=False)
plt.ylabel('Normalized % time in SZ')
plt.ylim(0,2)

# %% open loop speed
plt.figure()
ax0=plt.subplot(2,2,1)
plt.title('PBS in GPe')
pbs=np.array([(('10x' in x) and ('pbs' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[pbs,'speed']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='speed',ax=ax0,
                         clip_method=False)
plt.ylabel('Normalized speed (cm/s)')
plt.box(on=None)

ax=plt.subplot(2,2,2,sharey=ax0)
plt.title('Muscimol in GPe')
use=np.array([(('10x' in x) and ('muscimol' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[use,'speed']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='speed',ax=ax,
                         clip_method=False)
plt.ylabel('Normalized speed (cm/s)')
plt.ylim(0,3)
plt.box(on=None)


#Not normalized:
ax1=plt.subplot(2,2,3)
subset=np.stack(list(data.loc[pbs,'speed']),axis=0)
clip={'speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='speed',ax=ax1,
                         clip_method=False)
plt.ylabel('Speed (cm/s)')
plt.box(on=None)

ax=plt.subplot(2,2,4,sharey=ax1)
subset=np.stack(list(data.loc[use,'speed']),axis=0)
clip={'speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='speed',ax=ax,
                         clip_method=False)
plt.ylabel('Speed (cm/s)')
plt.ylim(0,15)
plt.box(on=None)

# %% amb_count
plt.figure()
ax0=plt.subplot(2,2,1)
plt.title('PBS in GPe')
pbs=np.array([(('10x' in x) and ('pbs' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[pbs,'amb_count']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'amb_count':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_count',ax=ax0,
                         clip_method=False)
plt.ylabel('Norm. amb. bouts')
plt.box(on=None)

ax=plt.subplot(2,2,2,sharey=ax0)
plt.title('Muscimol in GPe')
use=np.array([(('10x' in x) and ('muscimol' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[use,'amb_count']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'amb_count':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_count',ax=ax,
                         clip_method=False)
plt.ylabel('Norm. amb. bouts')
plt.ylim(0,3)
plt.box(on=None)


#Not normalized:
ax1=plt.subplot(2,2,3)
subset=np.stack(list(data.loc[pbs,'amb_count']),axis=0)
clip={'amb_count':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_count',ax=ax1,
                         clip_method=False)
plt.ylabel('Amb. bouts')
plt.box(on=None)

ax=plt.subplot(2,2,4,sharey=ax1)
subset=np.stack(list(data.loc[use,'amb_count']),axis=0)
clip={'amb_count':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_count',ax=ax,
                         clip_method=False)
plt.ylabel('Amb. bouts')
plt.ylim(0,4)
plt.box(on=None)

# %% amb speed
plt.figure()
ax0=plt.subplot(2,2,1)
plt.title('PBS in GPe')
pbs=np.array([(('10x' in x) and ('pbs' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[pbs,'amb_speed']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'amb_speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_speed',ax=ax0,
                         clip_method=False)
plt.ylabel('Norm. amb. speed')
plt.box(on=None)

ax=plt.subplot(2,2,2,sharey=ax0)
plt.title('Muscimol in GPe')
use=np.array([(('10x' in x) and ('muscimol' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[use,'amb_speed']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'amb_speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_speed',ax=ax,
                         clip_method=False)
plt.ylabel('Norm. amb. speed')
plt.ylim(0,3)
plt.box(on=None)


#Not normalized:
ax1=plt.subplot(2,2,3)
subset=np.stack(list(data.loc[pbs,'amb_speed']),axis=0)
clip={'amb_speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_speed',ax=ax1,
                         clip_method=False)
plt.ylabel('Amb. Speed (cm/s)')
plt.box(on=None)

ax=plt.subplot(2,2,4,sharey=ax1)
subset=np.stack(list(data.loc[use,'amb_speed']),axis=0)
clip={'amb_speed':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='amb_speed',ax=ax,
                         clip_method=False)
plt.ylabel('Amb. Speed (cm/s)')
plt.ylim(0,15)
plt.box(on=None)