#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:38:20 2021

@author: brian
"""
import pandas as pd
import json 
import numpy as np
from pathlib import Path
from gittislab import dataloc, ethovision_tools, signals, plots
import matplotlib.pyplot as plt



basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10/'
inc=[['AG5477_4']]
# basepath = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Right/5x30/'
# inc=[['AG4486_1']]
exc=[['exclude']]
ethovision_tools.unify_raw_to_csv(basepath,inc,exc)
ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,force_replace=True)
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
raw,meta=ethovision_tools.csv_load(pns,method='preproc')
ethovision_tools.boris_prep(basepath,inc,exc,plot_cols=['time','mouse_height','vel'], 
                            event_col='rear',event_thresh=0.5, method='preproc')

dlc=ethovision_tools.add_dlc_helper(raw,meta,pns.parent,force_replace = True)
dlc=dlc[0]
mouse_height=(dlc['dlc_rear_centroid_y']-dlc['dlc_front_centroid_y'])
head_xy=np.array(dlc.loc[:,('dlc_front_centroid_x','dlc_front_centroid_y')])
tail_xy=np.array(dlc.loc[:,('dlc_rear_centroid_x','dlc_rear_centroid_y')])
x=raw['x']
mh=signals.scale_per_dist(x,head_xy,tail_xy,mouse_height,step=2,poly_order = 2)
mhh=signals.max_normalize_per_dist(x,mouse_height,step=2,poly_order=2)
plt.figure()
plt.plot(raw['time'],mh/100)
plt.plot(raw['time'],mhh)

# %% Plot some questionable frames [update: they look good!]:

parts=['dlc_front_centroid','dlc_rear_centroid',]
part_colors = ['.c','.y']
framesets=[]
on,off=signals.thresh(mh,30, sign='Pos')
for o,f in zip(on,off):
    if o > f:
        print('o > f')
    m=round((o + f)/2)
    framesets.append([o,m,f])
# framesets=[[8349,8359,8369],[6327, 6337, 6347],[6159,6169,6179],[3356, 3366, 3376]]
plot_points=plots.gen_sidecam_plot_points(dlc,parts,framesets)

plots.etho_check_sidecamera(dataloc.video(pns.parent),framesets,
                            plot_points=plot_points,part_colors=part_colors)

# %% Different rear approach scaling mouse_height to body len
x = raw['x'].values


plt.figure()
plt.plot(raw['time'],mouse_height,'k')
plt.plot(raw['time'],mh,'--r')
# %%

#Load in human scoring:
fn_ka = pns.parent.joinpath('Rearing Observations.boris')
f = open(fn_ka,'r')
human=  json.loads(f.read() )
      
#Load in DLC scoring:
project_name = pns.parent.parts[-1]
fn = pns.parent.joinpath(project_name +'.boris')
f = open(fn,"r")
dlc= json.loads(f.read())

dlc_evts = dlc['observations']['event_check']['events']
for key in human['observations'].keys():
    if meta['anid'][0] in key:
        use_key=key
ka_evts = human['observations'][use_key]['events']

dlc_rear=raw['rear'].values.astype(bool)
ka_rear=np.zeros(dlc_rear.shape).astype(bool)
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


hit_rate = sum((dlc_rear==1) & (ka_rear==1)) / sum(ka_rear==1)
miss_rate = sum((dlc_rear == 0) & (ka_rear ==1)) / sum(ka_rear==1)
fa_rate= sum( (dlc_rear == 1) & (ka_rear == 0)) / sum(ka_rear==0)
print('Raw: %1.3f Hit, %1.3f Miss, %1.3f FA ' % (hit_rate, miss_rate, fa_rate))

#Per event, detect hits: 
hit=[]
for i in range(0,len(ka_evts)):
    if ka_evts[i][2] == 'a':
        k = i
        while 'd' not in ka_evts[k][2]:
            k += 1 
        start=ka_evts[i][0]
        stop=ka_evts[k][0]
        ind= (raw['time'] >=start ) & (raw['time'] < stop)
        if any(ind & dlc_rear):
            hit.append(1)
        else:
            hit.append(0)

# Detect false alarms
fa=[]
for i in range(0,len(dlc_evts)):
    if (dlc_evts[i][2] == 'start_rear'):
        k = i
        while ('stop_rear' not in dlc_evts[k][2]):
            k += 1 
        start=dlc_evts[i]
        stop=dlc_evts[k]
        ind= (raw['time'] >=start[0] ) & (raw['time'] <= stop[0])
        if any((ind == 1) & (ka_rear==1)):
            fa.append(0)
        else:
            fa.append(1)

e_hit_rate=sum(hit)/len(hit)
true_negative=len(hit) # This can be made to be true by adding confirmed true negative events (see below)
e_fa_rate=sum(fa)/true_negative
total_accuracy = (sum(hit) + sum(np.array(fa) == 0)) / (len(hit) + true_negative)
print('Event: Hit: %1.3f, FA: %1.3f, Accuracy = %1.3f' % (e_hit_rate, e_fa_rate, total_accuracy))

