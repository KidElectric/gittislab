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
x_s=raw['x']
y_s=raw['y']
cross=np.concatenate(([0],np.diff(raw['iz1'].astype(int)) > 0)).astype('bool')
cross_zero=np.median(x_s[cross])
print(cross_zero)
plt.figure()
plt.plot(x_s,y_s)
plt.plot(x_s-cross_zero,y_s)

# %% Plot approaches to zone 1:
c,nc=behavior.z1_to_z2_cross_detect(raw,meta)
plt.figure()
plt.plot([-25,-25,25,25,-25],[-25,25,25,-25,-25],'k')
plt.plot([0,0],[-25,25],'--k')
for i,j in c:
    fullx=raw['x'][i:j]
    plt.plot(fullx,raw['y'][i:j],10,c='r')

for i,j in nc:
    fullx=raw['x'][i:j]
    plt.plot(fullx,raw['y'][i:j],10,c='c')
    
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