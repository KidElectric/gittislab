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
pns=dataloc.raw_csv(basepath,inc[0],ex0)
raw,meta=ethovision_tools.csv_load(pns[0],method='preproc' )
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
# %%
data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'room','amb_meander','amb_bouts'])
temp=data
min_bout=1
use_dlc=False
keep_enter=[]
for ii,ee in zip(inc,exc):
    pns=dataloc.raw_csv(basepath,ii,ee)
    for pn in pns:
        temp={}
        raw,meta=ethovision_tools.csv_load(pn,method='preproc')
        temp['anid']=meta['anid'][0]
        temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                                 meta['stim_area'][0],
                                                 meta['opsin_type'][0])
        temp['proto']=meta['protocol'][0]
        temp['room']=meta.exp_room_number[0]
        stim_dur = round(np.mean(meta['stim_dur']))        
        vel_clip=behavior.stim_clip_grab(raw,meta,
                                          y_col='vel', 
                                          stim_dur=stim_dur)    
        
        enter=np.concatenate(([0],np.diff(raw['iz1'].astype(int)) > 0)).astype('bool')
        exit=np.concatenate(([0],np.diff(raw['iz1'].astype(int)) < 0)).astype('bool')
        dir = raw['dir']
        b=dir[enter]
        bb= np.array(raw['vel'].values[enter]) > 10
        b=b[bb]
        mb=circmean(b,high=180,low=-180)
        # mb=mode([x for x in b])
        keep_enter.append(mb)
        c=dir[exit]
        mc=circmean(c,high=180,low=-180)
        print(mc)
        # clip_ave=behavior.stim_clip_average(vel_clip)   
        
        # #### Calculate stim-triggered %time mobile:
        # percentage = lambda x: (np.nansum(x)/len(x))*100
      
        
        # #### Calculate ambulation bout properties:
        # raw['stim_zone'] = (raw['ambulation']==True) & (raw['vel']>5)
        # if any(raw['run']):
        #     amb_bouts=behavior.bout_analyze(raw,meta,'run',
        #                                     stim_dur=stim_dur,
        #                                     min_bout_dur_s=min_bout,
        #                                     use_dlc=use_dlc)
        #     temp['amb_meander']=np.nanmean(amb_bouts['meander'],axis=0)
        #     temp['amb_bouts']=np.nanmean(amb_bouts['rate'],axis=0)
        # else:
        #     temp['amb_meander']=[np.nan, np.nan, np.nan]
        #     temp['amb_bouts']=[0,0,0]
        # #### Calculate immobile bout properties:
        # im_bouts=behavior.bout_analyze(raw,meta,'im',
        #                                 stim_dur=stim_dur,
        #                                 min_bout_dur_s=min_bout,
        #                                 use_dlc=use_dlc)

# %% Take center crossing trajectory clips, align to crossing point

# %% Where are rears happening during zone task?