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

# fn1 = '/home/brian/Documents/rear_scoring.boris'
# fn = '/home/brian/Documents/template.boris'
fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/KA_rearing_observations.boris'
# reading the data from the file 
with open(fn) as f: 
    data = f.read() 
       
# reconstructing the data as a dictionary 
js = json.loads(data) 
  
f.close()

# with open(fn1) as f:
#     data=f.read()
# js1 = json.loads(data)

# %% Ethovision function version of boris integration:
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/'
inc=[['AG5362_3']]
exc=[['exclude']]
boris,raw,meta=ethovision_tools.boris_prep(basepath,inc,exc,plot_cols=['time','mouse_height','vel'], 
                            event_col='rear',event_thresh=0.5, method='preproc')

# fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/KA_rearing_observations.boris'

#Confirmed True negatives:
fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/dlc_and_ka_rearing_obs_with_fa_v3.boris'
# reading the data from the file 
with open(fn) as f: 
    data = f.read() 
       
# reconstructing the data as a dictionary 
js = json.loads(data) 
  
f.close()

dlc_evts = boris['observations']['event_check']['events']
ka_evts = js['observations']['AG5362_3_BIO22520']['events']
recoded=[]
for evt in ka_evts:
    if 'Recoded' in evt[4]:
        recoded.append(evt)
# %%
# KA vs. DLC event overlaps

ka_rear=np.zeros(raw['rear'].values.shape).astype(bool)
dur=[]
for i in range(0,len(ka_evts)):
    if ka_evts[i][2] == 'a':
        k = i
        while 'd' not in ka_evts[k][2]:
            k +=1 
        start=ka_evts[i][0]
        stop=ka_evts[k][0]
        dur.append(stop-start)
        ind= (raw['time'] >=start ) & (raw['time'] < stop)
        ka_rear[ind]=True
dlc_rear=raw['rear'].values.astype(bool)

hit_rate = sum((dlc_rear==1) & (ka_rear==1)) / sum(ka_rear==1)
miss_rate = sum((dlc_rear == 0) & (ka_rear ==1)) / sum(ka_rear==1)
fa_rate= sum( (dlc_rear == 1) & (ka_rear == 0)) / sum(ka_rear==0)
print('Raw: %1.3f Hit, %1.3f Miss, %1.3f FA ' % (hit_rate,miss_rate,fa_rate))

#Per event, detect hits: 
hit=[]
for i in range(0,len(ka_evts)):
    if ka_evts[i][2] == 'a':
        k = i
        while 'd' not in ka_evts[k][2]:
            k +=1 
        start=ka_evts[i][0]
        stop=ka_evts[k][0]
        ind= (raw['time'] >=start ) & (raw['time'] < stop)
        if any(ind & dlc_rear):
            hit.append(1)
        else:
            hit.append(0)

# Detect false alarms
fa=[]
for i in range(0,len(dlc_evts),2):
    start=dlc_evts[i]
    stop=dlc_evts[i+1]
    ind= (raw['time'] >=start[0] ) & (raw['time'] <= stop[0])
    if any(ind & ka_rear):
        fa.append(0)
    else:
        fa.append(1)
        start[2]='fa_rear_onset'
        stop[2]='fa_rear_offset'
        dlc_evts[i]=start
        dlc_evts[i+1]=stop
        
e_hit_rate=sum(hit)/len(hit)
true_negative=len(hit) # This can be made to be true by adding confirmed true negative events (see below)
e_fa_rate=sum(fa)/true_negative
print('Event: Hit: %1.3f, FA: %1.3f' % (e_hit_rate, e_fa_rate))


add_events = true_negative - sum(fa)
not_rear= ~ (ka_rear | dlc_rear)
on,off=signal.thresh(not_rear,0.5,'Pos')
onset=raw['time'][on].values
offset=raw['time'][off].values
if offset[0] < onset[0]:
    offset=offset[1:]
    
# plt.figure(),plt.plot(raw['time'],not_rear),plt.plot(onset,np.ones((len(onset),1)),'ro'),plt.plot(offset,np.ones((len(offset),1)),'bo')
added=0
rear_dur=np.mean(dur)
new_evt=[]
# while added < add_events:
add_fa_events = False
if add_fa_events == True:
    for i,j in zip(onset,offset):
        if ((j-i)> rear_dur*3 ) and (added < add_events):
            dlc_evts.append([i+rear_dur,meta['anid'][0],'fa_rear_onset','',''])
            dlc_evts.append([i+(rear_dur*2),meta['anid'][0],'fa_rear_offset','',''])
            added += 1
            new_end =i+(rear_dur*2)
            if (j- new_end) > (rear_dur*3) and (added < add_events): #if there is space, add another one:
                dlc_evts.append([new_end+rear_dur,meta['anid'][0],'fa_rear_onset','',''])
                dlc_evts.append([new_end+(rear_dur*2),meta['anid'][0],'fa_rear_offset','',''])
                added += 1

# %% Integrating KA w/ DLC:
# Go into KA observation and add DLC events:

ka_evts = js['observations']['AG5362_3_BIO22520']['events']
#Remove previous DLC from ka if necessary:
keep = []
for evt in ka_evts:
    if ('a' == evt[2]) or ('d' == evt[2]) or ('s' == evt[2]):
        keep.append(evt)
combo= dlc_evts + keep

# Need to sort by timing:
temp = pd.DataFrame(combo).sort_values(by=0)

# js['observations']['AG5362_3_BIO22520']['events'] = temp.values.tolist()
boris['observations']['event_check']['events']=  temp.values.tolist()
new_fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/dlc_and_ka_rearing_confirmed_fa_final.boris'
f = open(new_fn,"w")
json.dump(boris, f, sort_keys=True, indent=4)
f.close()

# %% Load TRUE NEGATIVE CONFIRMED boris file:
fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/dlc_and_ka_rearing_confirmed_fa_final_v2.boris'
f = open(fn,'r')
fa_conf=  json.loads(f.read() )
       
fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/dlc_and_ka_rearing_obs_with_fa.boris'
f = open(fn,"r")
dlc_ka= json.loads(f.read())

dlc_ka_v1_evts = dlc_ka['observations']['AG5362_3_BIO22520']['events']
dlc_ka_v2_evts = fa_conf['observations']['event_check']['events']

#Count nrear pre-post:
pre_hit = 0
for i in range(0,len(dlc_ka_v1_evts)):
    if dlc_ka_v1_evts[i][2] == 'a':
        pre_hit +=1

post_hit = 0 

for i in range(0,len(dlc_ka_v2_evts)):
    if dlc_ka_v2_evts[i][2] == 'a':
        post_hit +=1

print('Before 2nd round: %d rears (KA), post: %d rears (+DLC, Confirmed by KA)' % (pre_hit,post_hit))

ka_rear=np.zeros(raw['rear'].values.shape).astype(bool)
dur=[]
for i in range(0,len(dlc_ka_v2_evts)):
    if dlc_ka_v2_evts[i][2] == 'a':
        k = i
        while 'd' not in dlc_ka_v2_evts[k][2]:
            k +=1 
        start=dlc_ka_v2_evts[i][0]
        stop=dlc_ka_v2_evts[k][0]
        dur.append(stop-start)
        ind= (raw['time'] >=start ) & (raw['time'] < stop)
        ka_rear[ind]=True
dlc_rear=raw['rear'].values.astype(bool)

hit_rate = sum((dlc_rear==1) & (ka_rear==1)) / sum(ka_rear==1)
miss_rate = sum((dlc_rear == 0) & (ka_rear ==1)) / sum(ka_rear==1)
fa_rate= sum( (dlc_rear == 1) & (ka_rear == 0)) / sum(ka_rear==0)
print('Raw: %1.3f Hit, %1.3f Miss, %1.3f FA ' % (hit_rate, miss_rate, fa_rate))

#Per event, detect hits: 
hit=[]
for i in range(0,len(dlc_ka_v2_evts)):
    if dlc_ka_v2_evts[i][2] == 'a':
        k = i
        while 'd' not in dlc_ka_v2_evts[k][2]:
            k += 1 
        start=dlc_ka_v2_evts[i][0]
        stop=dlc_ka_v2_evts[k][0]
        ind= (raw['time'] >=start ) & (raw['time'] < stop)
        if any(ind & dlc_rear):
            hit.append(1)
        else:
            hit.append(0)

# Detect false alarms
fa=[]
for i in range(0,len(dlc_ka_v2_evts)):
    if dlc_ka_v2_evts[i][2] == 'start_rear':
        k = i
        while 'stop_rear' not in dlc_ka_v2_evts[k][2]:
            k += 1 
        start=dlc_ka_v2_evts[i]
        stop=dlc_ka_v2_evts[k]
        ind= (raw['time'] >=start[0] ) & (raw['time'] <= stop[0])
        if any((ind == 1) & (ka_rear==1)):
            fa.append(0)
        else:
            fa.append(1)

e_hit_rate=sum(hit)/len(hit)
true_negative=len(hit)-1 # This can be made to be true by adding confirmed true negative events (see below)
e_fa_rate=sum(fa)/true_negative
total_accuracy = (sum(hit) + sum(np.array(fa) == 0)) / (len(hit) + true_negative)
print('Event: Hit: %1.3f, FA: %1.3f, Accuracy = %1.3f' % (e_hit_rate, e_fa_rate, total_accuracy))




# n_negative = 
# Detect false alarms
# fa=[]
# for i in range(0,len(dlc_evts),2):
#     start=dlc_evts[i]
#     stop=dlc_evts[i+1]
#     ind= (raw['time'] >=start[0] ) & (raw['time'] < stop[0])
#     if any(ind & ka_rear):
#         fa.append(0)
#     else:
#         fa.append(1)
#         start[2]='fa_rear_onset'
#         stop[2]='fa_rear_offset'
#         dlc_evts[i]=start
#         dlc_evts[i+1]=stop
        
# %% Raw script code:

#Enter name of new event type to observe:
obs_name='dlc_rear_detect' #Can have multiple observation types per file
project_name = 'AG6382_4_BI020421' #Used in saving file: project_name.boris
#Identify folder containing Raw*.csv file from noldus and original video:
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
expfolder='Str/Naive/A2A/Ai32/Bilateral/10x10_gpe_pbs/' + project_name + '/'
video_fn=dataloc.video(basepath+expfolder) #or manually: basepath + expfolder + 'Trial    257.mpg'
if not isinstance(video_fn,Path):
    pnfn=Path(video_fn) 
else:
    pnfn=video_fn


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
js['observations'][obs_name]['file']['1'][0] = str(video_fn)

fn='/home/brian/Documents/dlc_rearing.boris'

#Load Raw ethovision tracking or other .csv to use for new event
raw_pnfn=dataloc.raw_csv(pnfn.parent) #Locate raw_pnfn .csv file
raw,meta=ethovision_tools.csv_load(raw_pnfn,method='preproc')

#If desired, include a copy of this or other columns to plot in boris:
temp_fn=meta['pn'][0].parent.joinpath('boris_viz.csv')
temp=raw.loc[:,('time','mouse_height','vel')] #columns 2,3,4 for boris purposes
# temp['rear']=temp['rear'] * meta['rear_thresh'][0]
vel_max=50
for col in temp.columns:
    isnan=np.isnan(temp[col])
    if any(isnan):
        temp.loc[isnan,col]=0 #For plotting purposes replace with 0
    
    if ('vel' in col):
        mm=(temp[col] > vel_max)
        temp.loc[mm,col]= 0 #For plotting purposes eliminate velocity artifacts
        
temp.to_csv(temp_fn, sep ='\t')
js['subjects_conf']={'0': {'key': 'q', 'name': meta['anid'][0], 'description': ''}}

#Add information on which data to plot from this temp file (boris will show this
# sliding along with video to cross-check... useful to use continuous trace that event
#-detection is based off of)
js['observations'][obs_name]['plot_data']={"0":{'file_path': str(temp_fn), #Raw signal to detect events off of
                                                'columns': '2,3', #must always indicate time column and trace column by index sep by comma
                                                'title' : 'Norm. mouse height (px)', #Normalized mouse height (px)
                                                "variable_name": "", 
                                                "converters": {}, 
                                                "time_interval": "60",
                                                "time_offset": "0",
                                                "substract_first_value": "True",
                                                "color": "b-"},
                                           
                                           "1":{'file_path': str(temp_fn), #rear detect at thresh
                                                'columns': '2,4', #must always indicate time column and trace column by index sep by comma
                                                'title' : 'Norm. speed (cm/s)',
                                                "variable_name": "", 
                                                "converters": {}, 
                                                "time_interval": "60",
                                                "time_offset": "0",
                                                "substract_first_value": "True",
                                                "color": "g-"},
                                           }

#Update media info to reflect video of interest (many can be added but just one shown here):
med_info=js['observations'][obs_name]['media_info']
for key in js['observations'][obs_name]['media_info'].keys():
    if 'length' == key:
        med_info[key]={str(video_fn) : raw['time'].values[-1]}
    if 'fps' == key:
        med_info[key]={str(video_fn) : round(meta['fs'][0],ndigits=2)}
    if 'hasVideo' == key:
        med_info[key]={str(video_fn) : True}
    if 'hasAudio' == key:
        med_info[key] = {str(video_fn) : False}
js['observations'][obs_name]['media_info'] = med_info

#Collect event onset and offset times in video from a custom data column:
on,off = signal.thresh(raw['rear'],0.5)
on_times=raw['time'].values[on]
off_times=raw['time'].values[off]

event_times=[on_times,off_times]
event_names=['rear_start_dlc','rear_stop_dlc'] #Custom event type names
new_evt=[]
for on,off in zip(on_times,off_times):
    new_evt.append([on,meta['anid'][0],event_names[0],'',''])
    new_evt.append([off,meta['anid'][0],event_names[1],'',''])
js['observations'][obs_name]['events']=new_evt


fn='/home/brian/Documents/%s.boris' % project_name

#Save this new file:
f = open(fn,"w")
json.dump(js, f, sort_keys=True, indent=4)
f.close()
print('Saved.')
# %% Plot
rear_thresh=0.6
mouse_height=dlc['dlc_front_over_rear_length']
start_peak,stop_peak = signal.thresh(mouse_height,rear_thresh,'Pos')
# peaks,start_peak,stop_peak = signal.expand_peak_start_stop(mouse_height,height=rear_thresh,min_thresh=min_thresh)
rear=np.zeros(mouse_height.shape)
for start,stop in zip(start_peak,stop_peak):
    rear[start:stop]=1
    
# %% Examine rearing at different or same threshold:
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
    
# %% Ethovision function version
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10_gpe_pbs/'
inc=[['AG6382_4']]
exc=[['exclude']]
ethovision_tools.boris_prep(basepath,inc,exc,plot_cols=['time','mouse_height','vel'], 
                            event_col='mouse_height',event_thresh=0.5, method='preproc')

# %%
ka_score='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/KA_rearing_observations.boris'
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