#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:54:30 2021
Example of reading in .boris file as a dictionary

@author: brian
"""

import pandas as pd
import json 
import numpy as np
from pathlib import Path
from gittislab import dataloc, ethovision_tools,signal
import matplotlib.pyplot as plt

fn = '/home/brian/Documents/rear_scoring.boris'
# reading the data from the file 
with open(fn) as f: 
    data = f.read() 
  
print("Data type before reconstruction : ", type(data)) 
      
# reconstructing the data as a dictionary 
js = json.loads(data) 
  
print("Data type after reconstruction : ", type(js)) 
print(js) 
js['observations']['new']=js['observations']['test']
f.close()

# %%
obs_name='dlc_rear_detect'
js['observations'][obs_name]=js['observations']['test']
js['observations'].pop('test')
original_video=Path(js['observations'][obs_name]['file']['1'][0])

fn='/home/brian/Documents/dlc_rearing.boris'

#Add rear events detected by deeplabcut!
raw_pnfn=dataloc.raw_csv(original_video.parent)
raw,meta=ethovision_tools.csv_load(raw_pnfn,method='preproc')

on,off = signal.thresh(raw['rear'],0.5)
on_times=raw['time'].values[on]
off_times=raw['time'].values[off]

event_times=[on_times,off_times]
event_names=['rear_start_dlc','rear_stop_dlc']
new_evt=[]
for on,off in zip(on_times,off_times):
    new_evt.append([on,'',event_names[0],'',''])
    new_evt.append([off,'',event_names[1],'',''])
js['observations'][obs_name]['events']=new_evt

fn='/home/brian/Documents/dlc_rearing_3.boris'
#Find rear via 
f = open(fn,"w")
json.dump(js, f, sort_keys=True, indent=4)
f.close()
print('Saved.')
# %%
rear_thresh=0.6
mouse_height=dlc['dlc_front_over_rear_length']
start_peak,stop_peak = signal.thresh(mouse_height,rear_thresh,'Pos')
# peaks,start_peak,stop_peak = signal.expand_peak_start_stop(mouse_height,height=rear_thresh,min_thresh=min_thresh)
rear=np.zeros(mouse_height.shape)
for start,stop in zip(start_peak,stop_peak):
    rear[start:stop]=1
    
# %% Examine rearing thresholding:
dlc, meta = ethovision_tools.add_dlc_helper(raw,meta,
                                     raw_pnfn.parent,
                                     force_replace=True,
                                     rear_thresh=0.6,
                                     min_thresh = 0.55)
plt.figure()
x=raw['time'].values/60
oo,ff= signal.thresh(dlc['dlc_front_over_rear_length'],0.6)
plt.plot(x,dlc['dlc_front_over_rear_length'],'k')
plt.plot(x,raw['rear'])
# plt.plot(x,dlc['dlc_is_rearing_logical'])
plt.plot(x[oo],0.5*np.ones(oo.shape),'og')
plt.plot(x[ff],0.5*np.ones(oo.shape),'or')
plt.plot([0,x[-1]], meta['rear_thresh'][0:2],'--g')
plt.plot([0,x[-1]], meta['rear_min_thresh'][0:2],'--r')                    
#%% Locating events in the dictionary:

events = js['observations']['test']['events']
for e in events:
    a=['rearstop' in str(x) for x in e]
    if any(a):
        print(e[0])

# %% 
with open(fn) as f:
    data = f.read()
js = json.loads(data)
print(js)
f.close()

# %% Rerun processing of rear detection on subset:
    
ex0=['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG']]
make_preproc = True
exc=[ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/'
ethovision_tools.raw_csv_to_preprocessed_csv(basepath,
                                             inc,exc,force_replace=True)

summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))