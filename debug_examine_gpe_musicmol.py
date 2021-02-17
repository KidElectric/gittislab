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
import time

# %% Process any new files:
ex0=['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','10x10_gpe_pbs'],['AG','10x10_gpe_muscimol'],['60min_gpe_muscimol']]
make_preproc = True
exc=[ex0,ex0,ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath, inc, exc,
                                  force_replace=False,
                                  win=10)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,
                                             force_replace=False,win=10)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc) 
   
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% 10x10 1mW w/ PBS in GPe
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','mW','mw']

# inc=[['AG','10x10_gpe_muscimol',],]

inc=[['AG','10x10_gpe_pbs',],]
exc=[ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

for ii,ee in zip(inc,exc):
    pns=dataloc.raw_csv(basepath,ii,ee)
    
    for pn in pns:
        temp={}
        raw,meta=ethovision_tools.csv_load(pn,method='preproc')
        plots.plot_openloop_day(raw,meta,save=True, close = False)
        

# %% Combined openloop day summary:
inc=[['AG','10x10_gpe_pbs',],]
exc=[['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','mW','mw','AG6343']]
data = behavior.open_loop_summary_collect(basepath,[inc[0]],[exc[0]])
# %%
fig=plots.plot_openloop_mouse_summary(data)

#%% Compare 60min after muscimol mobility vs. time w/ 10x10 Str mobility thresh:
ex0=['exclude','Other XLS','Exclude','AG6343','AG6382_2']
inc=[['10x10_gpe_pbs'],['60min_gpe_muscimol']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
exc=[ex0,ex0]

percentage = lambda x: round((np.nansum(x)/len(x))*100,ndigits=2)
count=[x for x in range(len(inc))]
for i,ii,ee in zip(count,inc,exc):
    if i == 0:
        data = behavior.open_loop_summary_collect(basepath,[ii],[ee])
# %% CONTINUATION FROM ABOVE CELL-->
ee=ex0
ii=['60min_gpe_muscimol']
pns=dataloc.raw_csv(basepath,ii,ee)
cont=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'mobility','time','thresh'])
temp={}
for pn in pns:
    raw,meta=ethovision_tools.csv_load(pn,columns=['time','m','vel'])
    temp['anid']=meta['anid'][0]
    temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                             meta['stim_area'][0],
                                             meta['opsin_type'][0])
    temp['proto']=meta['protocol'][0]
    x,y =signal.bin_analyze(raw['time'].values,raw['m'].values,
                            bin_dur=(60*5),fun = percentage)
    # x,y =signal.bin_analyze(raw['time'].values,raw['vel'].values,
    #             bin_dur=60,)
    
    x=x/60
    ind = data['anid']==temp['anid']
    temp['time']=x
    temp['mobility']=y
    thresh = data.loc[ind,'per_mobile'].values[0][1] #Value to compare w/ continuous
    # thresh = data.loc[ind,'stim_speed'].values[0][0][1]
    temp['thresh']=thresh
    plt.figure()
    plt.plot(x,y,'k')
    plt.title(temp['anid'])
    plt.plot([0,x[-1]],np.ones((2,1))*thresh,'--r',label='Str A2a ChR2')
    plt.xlabel('Time from muscimol infusion (min)')
    plt.ylabel('%Time mobile')
    plt.legend()
    cont=cont.append(temp,ignore_index=True)
# %% AVERAGE ABOVE:
yy=np.vstack(cont['mobility'])
y=np.mean(yy,axis=0)
x=cont['time'][0]
thresh=np.mean(cont['thresh'])
plt.figure()
plt.plot(x,y,'k')
plt.plot([0,x[-1]],np.ones((2,1))*thresh,'--r',label='Str A2a ChR2')
plt.xlabel('Time from GPe muscimol infusion (0.25ul) (min)')
plt.ylabel('%Time mobile')
plt.legend()

# %% Look at mouse speed vs. time in each case:
    
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','mW','mw']

inc=[['AG','zone_','AG6343_4'],
     ['AG','zone_','AG6343_5'],]
exc=[ex0,ex0,ex0,ex0]

basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'room','speed','time'])
temp=data
min_bout=1
use_dlc=False
keep_enter=[]
plt.close('all')
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

        smth_amount=round(60*meta['fs'][0])
        speed=raw['vel'].values.astype(float)
        speed[0:10]=speed[10]
        speed = signal.boxcar_smooth(speed,smth_amount)
        temp['speed']=speed
        temp['time']=raw['time']/60
        data=data.append(temp,ignore_index=True)
# %%
a=data['anid'].unique()
for an in a:
    plt.figure()
    plt.title('%s' % an)
    for i,row_an in enumerate(data['anid']):
        if row_an == an:
            speed=data.loc[i,'speed']
            plt.plot(data.loc[i,'time'],speed)

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
            # plots.plot_openloop_day(raw,meta)
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
h0=plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='in_sz',ax=ax,
                         clip_method=False)
plt.ylabel('% time in SZ')


ax=plt.subplot(1,2,1,sharey=ax)
plt.title('No muscimol in GPe')
zone=np.array([(('zone' in x) and not ('muscimol' in x)) for x in data['proto']])
subset=np.stack(list(data.loc[zone,'in_sz']),axis=0)
subset=np.array([x/x[0] for x in subset])
clip={'in_sz':subset}
_,h1=plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='in_sz',ax=ax,
                         clip_method=False)
plt.legend(h1,data['anid'].unique()[[1,0]],loc = 'upper left')
plt.ylabel('Normalized % time in SZ')
plt.ylim(0,4)

#%% Plot a normalized change PBS -> Musc. AG6343_4 
m=np.array(data['anid'])=='AG6343_4'
# m=np.array(data['anid'])=='AG6343_5'
dsub=data.iloc[m,:]
out=[x for x in range(4)]
olfield='mobile'
for index, row in dsub.iterrows():
    z='zone' in row['proto']
    musc = 'muscimol' in row['proto']
    pbs = 'pbs' in row['proto']
    if (not z) and pbs:
        out[0] = row[olfield][1]/row[olfield][0] * 100
    elif (not z) and musc:
        out[1] = row[olfield][1]/row[olfield][0] * 100
    elif (z) and (not musc):
        out[2] = row['in_sz'][1]/row['in_sz'][0] * 100
    elif z and musc:
        out[3] = row['in_sz'][1]/row['in_sz'][0] * 100
plt.figure(figsize=(5.07,5.37))
labels=('10x10 PBS','10x10 Musc.','Zone','Zone Musc.')
h=plt.bar(labels,out)
for i in range(2,4):
    h[i].set_facecolor('r')
plt.ylim(0,100)
plt.ylabel('% of baseline (Stim On / Stim Off)')
ind=slice(0,3,2)
plt.legend(h[ind],('% Time Mobile','% Time in SZ'),loc='upper left')

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