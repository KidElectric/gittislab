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
        c,nc=behavior.z1_to_z2_cross_detect(raw,meta)
        t=[i for i in range(4)]
        t[0]=0
        t[1]=meta.task_start[0] * meta['fs'][0]
        t[2]=meta.task_stop[0]* meta['fs'][0]
        t[3]=meta.exp_end[0]* meta['fs'][0]
        tot_c=np.zeros((2,3))

        for i in range(len(t)-1):
            in_period=0
            for cross in c:
                if (cross[0] >=t[i]) and (cross[1] < t[i+1]):
                    in_period += 1
            tot_c[0,i]=in_period
            in_period=0
            for cross in nc:
                if (cross[0] >=t[i]) and (cross[1] < t[i+1]):
                    in_period += 1
            tot_c[1,i]=in_period
        completed_cross=(tot_c[0,:]/np.nansum(tot_c,axis=0)) * 100
        temp['cross_per']= completed_cross
        temp['cross_counts']=np.nansum(tot_c,axis=0)
        data=data.append(temp,ignore_index=True)
# %% Where are rears happening during zone task?