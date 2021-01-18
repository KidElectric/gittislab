#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:36:52 2021

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

# %%
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude',
     '_gpe_muscimol','_gpe_pbs','mW','mw']
# inc=[['AG','GPe','CAG','Arch','10x']]
# exc=[ex0]
inc=[['AG','GPe','CAG','Arch','10x'],
     ['AG','Str','A2A','Ai32','10x'],
     ['AG','Str','A2A','ChR2','10x']]
exc=[ex0,ex0,ex0]

basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'amb_vel','amb_meander','amb_bouts'])
temp=data
min_bout=1
use_dlc=False
for ii,ee in zip(inc,exc):
    pns=dataloc.raw_csv(basepath,ii,ee)
    for pn in pns:
        temp={}
        raw,meta=ethovision_tools.csv_load(pn)
        temp['anid']=meta['anid'][0]
        temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                                 meta['stim_area'][0],
                                                 meta['opsin_type'][0])
        temp['proto']=meta['protocol'][0]
        stim_dur = round(np.mean(meta['stim_dur']))        
        vel_clip=behavior.stim_clip_grab(raw,meta,
                                          y_col='vel', 
                                          stim_dur=stim_dur)        
        clip_ave=behavior.stim_clip_average(vel_clip)   
        
        #### Calculate stim-triggered %time mobile:
        percentage = lambda x: (np.nansum(x)/len(x))*100
        m_clip=behavior.stim_clip_grab(raw,meta,y_col='m', 
                                        stim_dur=stim_dur, 
                                        summarization_fun=percentage)        
        
        #### Calculate ambulation bout properties:
        raw['run'] = (raw['ambulation']==True) & (raw['vel']>7)
        if any(raw['run']):
            amb_bouts=behavior.bout_analyze(raw,meta,'run',
                                            stim_dur=stim_dur,
                                            min_bout_dur_s=min_bout,
                                            use_dlc=use_dlc)
            temp['amb_meander']=np.nanmean(amb_bouts['meander'],axis=0)
            temp['amb_bouts']=np.nanmean(amb_bouts['speed'],axis=0)
        else:
            temp['amb_meander']=[np.nan, np.nan, np.nan]
            temp['amb_bouts']=[0,0,0]
        #### Calculate immobile bout properties:
        im_bouts=behavior.bout_analyze(raw,meta,'im',
                                        stim_dur=stim_dur,
                                        min_bout_dur_s=min_bout,
                                        use_dlc=use_dlc)

        data=data.append(temp,ignore_index=True)
        
# %% Note: try limiting "ambulation" to "running"
plt.figure()
ax=plt.subplot(1,2,1)
chr2= np.array([('Ai32'in x or 'ChR2' in x) for x in data['cell_area_opsin']])
subset=np.stack(list(data.loc[chr2,'amb_meander']),axis=0)
clip=amb_bouts
clip={'run_meander':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='run_meander', ax=ax,
                         clip_method=False)
plt.ylim(0,15)
# print(np.nanmean(subset,axis=0))

ax=plt.subplot(1,2,2)
subset=np.stack(list(data.loc[~chr2,'amb_meander']),axis=0)
clip={'run_meander':subset}
plots.mean_bar_plus_conf(clip, ['Pre','Dur','Post'],
                         use_key='run_meander',ax=ax,clip_method=False)
plt.ylim(0,15)
# print(np.nanmean(subset,axis=0))
# %% Can I reasonably combine Ai32 & ChR2 mice?

# %% What percent of time ethovision says immobile is animal rearing?