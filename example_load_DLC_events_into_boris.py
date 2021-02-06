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

fn1 = '/home/brian/Documents/rear_scoring.boris'
fn = '/home/brian/Documents/template.boris'
# reading the data from the file 
with open(fn) as f: 
    data = f.read() 
       
# reconstructing the data as a dictionary 
js = json.loads(data) 
  
f.close()

with open(fn1) as f:
    data=f.read()
js1 = json.loads(data)
# %%

#Enter name of new event type to observe:
obs_name='dlc_rear_detect' #Can have multiple observation types per file
project_name = 'test_template' #Used in saving file: project_name.boris
#Identify folder containing Raw*.csv file from noldus and original video:
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
expfolder='Str/Naive/A2A/Ai32/Bilateral/10x10_gpe_pbs/AG6382_3_BI020421/'
video_fn=basepath +expfolder + 'Trial   237.mpg'
pnfn=Path(video_fn)


#Import a template file to update with custom events:
fn = '/home/brian/Documents/template.boris'
# Read template.boris file:
with open(fn) as f: 
    data = f.read()   
# Reconstruct .boris file as a dictionary 
js = json.loads(data) 
f.close()

# Create a new observation using the template and the name given above:
js['observations'][obs_name]=js['observations']['new']
js['observations'].pop('new') #Remove template observation
js['observations'][obs_name]['file']['1'][0] = video_fn

fn='/home/brian/Documents/dlc_rearing.boris'

#Load Raw ethovision tracking or other .csv to use for new event
raw_pnfn=dataloc.raw_csv(pnfn.parent) #Locate raw_pnfn .csv file
raw,meta=ethovision_tools.csv_load(raw_pnfn,method='preproc')

#If desired, include a copy of this or other columns to plot in boris:
temp_fn=meta['pn'][0].parent.joinpath('boris_viz.csv')
temp=raw.loc[:,('time','mouse_height','rear')] #columns 2,3,4 for boris purposes
temp.to_csv(temp_fn, sep ='\t')

#Add information on which data to plot from this temp file (boris will show this
# sliding along with video to cross-check... useful to use continuous trace that event
#-detection is based off of)
js['observations'][obs_name]['plot_data']={"0":{'file_path': str(temp_fn),
                                                'columns': '2,3', #must always indicate time column and trace column by index sep by comma
                                                'title' : '',
                                                "variable_name": "", 
                                                "converters": {}, 
                                                "time_interval": "60",
                                                "time_offset": "0",
                                                "substract_first_value": "True",
                                                "color": "b-"}}

#Update media info to reflect video of interest (many can be added but just one shown here):
med_info=js['observations'][obs_name]['media_info']
for key in js['observations'][obs_name]['media_info'].keys():
    if 'length' == key:
        med_info[key]={video_fn : raw['time'].values[-1]}
    if 'fps' == key:
        med_info[key]={video_fn : round(meta['fs'][0],ndigits=2)}
    if 'hasVideo' == key:
        med_info[key]={video_fn : True}
    if 'hasAudio' == key:
        med_info[key] = {video_fn : False}
js['observations'][obs_name]['media_info'] = med_info

#Collect event onset and offset times in video from a custom data column:
on,off = signal.thresh(raw['rear'],0.5)
on_times=raw['time'].values[on]
off_times=raw['time'].values[off]

event_times=[on_times,off_times]
event_names=['rear_start_dlc','rear_stop_dlc'] #Custom event type names
new_evt=[]
for on,off in zip(on_times,off_times):
    new_evt.append([on,'',event_names[0],'',''])
    new_evt.append([off,'',event_names[1],'',''])
js['observations'][obs_name]['events']=new_evt


fn='/home/brian/Documents/%s.boris' % project_name

#Save this new file:
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