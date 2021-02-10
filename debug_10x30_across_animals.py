#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:36:52 2021

@author: brian
"""
from gittislab import plots, signal, behavior, dataloc, ethovision_tools, plots, profile_fun
import os
from pathlib import Path
import numpy as np
import pandas as pd
# import modin.pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem, t
import pdb
import math
import time
# %%
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude',
     '_gpe_muscimol','_gpe_pbs','mW','mw']
# inc=[['AG','GPe','CAG','Arch','10x']]
# exc=[ex0]
inc=[['AG','Str','A2A','Ai32','10x'],
     ['AG','Str','A2A','ChR2','10x']] # ['AG','GPe','CAG','Arch','10x']
exc=[ex0,ex0,ex0]

basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
data = behavior.open_loop_summary_collect(basepath,[inc[0]],[exc[0]])

# %%
fig=plots.plot_openloop_mouse_summary(data)
# %% FoxP2

ex0=['exclude','bad','Exclude','broken', 'Other XLS','Exclude',]
# inc=[['AG','GPe','CAG','Arch','10x']]
exc=[ex0]
inc=[['AG','GPe','FoxP2','ChR2','10x10_20mW'],]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10)
ethovision_tools.raw_csv_to_preprocessed_csv(basepath,
                                             inc,exc,force_replace=False,
                                             win=10)
for ii,ee in zip(inc,exc):
   pns=dataloc.raw_csv(basepath,ii,ee)
   for pn in pns:
       temp={}
       raw,meta=ethovision_tools.csv_load(pn,method='preproc' )
       plots.plot_openloop_day(raw,meta)
# %%

# def test_batch_analyze(inc,exc):

data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'amb_vel','amb_meander','amb_bouts','amb_directed'])
temp=data
min_bout=1
use_dlc=False
use_cols=['time','vel','im','dir','ambulation','meander']
for ii,ee in zip(inc,exc):
    pns=dataloc.raw_csv(basepath,ii,ee)
    for pn in pns:
        temp={}
        raw,meta=ethovision_tools.csv_load(pn,columns=use_cols,method='preproc' )
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
        raw['m']=~raw['im']
        m_clip=behavior.stim_clip_grab(raw,meta,y_col='m', 
                                        stim_dur=stim_dur, 
                                        summarization_fun=percentage)        
        
        #### Calculate ambulation bout properties:
        raw['run'] = (raw['ambulation']==True) & (raw['vel']>5)
        # raw['flight']=(raw['vel'] > (4* np.mean(raw['vel']))) #Flight, Yilmaz & Meister 2013
        raw['flight']=(raw['vel'] > (3 * np.mean(raw['vel'])))
        if any(raw['run']):
            amb_bouts=behavior.bout_analyze(raw,meta,'flight',
                                            stim_dur=stim_dur,
                                            min_bout_dur_s=min_bout,
                                            use_dlc=use_dlc)
            temp['amb_meander']=np.nanmean(amb_bouts['meander'],axis=0)
            temp['amb_directed']=np.nanmean(amb_bouts['directed'],axis=0)
            temp['amb_bouts']=np.nanmean(amb_bouts['rate'],axis=0)
        else:
            temp['amb_meander']=[np.nan, np.nan, np.nan]
            temp['amb_directed']=[np.nan, np.nan, np.nan]
            temp['amb_bouts']=[0,0,0]
        #### Calculate immobile bout properties:
        im_bouts=behavior.bout_analyze(raw,meta,'im',
                                        stim_dur=stim_dur,
                                        min_bout_dur_s=min_bout,
                                        use_dlc=use_dlc)

        data=data.append(temp,ignore_index=True)
        
# %% Flight
plt.figure()
ax=plt.subplot(1,2,1)
plt.title('Str A2a ChR2')
chr2= np.array([('Ai32'in x or 'ChR2' in x) for x in data['cell_area_opsin']])
subset=np.stack(list(data.loc[chr2,'amb_bouts']),axis=0)
clip={'run_meander':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='run_meander', ax=ax,
                         clip_method=False)
plt.ylabel('Flight bouts / s')
plt.ylim(0,0.3)


ax=plt.subplot(1,2,2)
plt.title('GPe CAG Arch')
subset=np.stack(list(data.loc[~chr2,'amb_bouts']),axis=0)
clip={'run_meander':subset}
plots.mean_bar_plus_conf(clip, ['Pre','Dur','Post'],
                         use_key='run_meander',ax=ax,
                         clip_method=False)
plt.ylim(0,0.3)
plt.ylabel('Flight bouts / s')

# %% Note: try limiting "ambulation" to "running"
plt.figure()
ax=plt.subplot(1,2,1)
plt.title('Str A2a ChR2')
chr2= np.array([('Ai32'in x or 'ChR2' in x) for x in data['cell_area_opsin']])
subset=np.stack(list(data.loc[chr2,'amb_directed']),axis=0)
clip=amb_bouts
clip={'run_meander':subset}
plots.mean_bar_plus_conf(clip,['Pre','Dur','Post'],
                         use_key='run_meander', ax=ax,
                         clip_method=False)
plt.ylabel('Directedness (cm/deg)')
plt.ylim(0,3)
subset=np.stack(list(data.loc[chr2,'amb_bouts']),axis=0)
print(np.nanmean(subset,axis=0))

ax=plt.subplot(1,2,2)
plt.title('GPe CAG Arch')
subset=np.stack(list(data.loc[~chr2,'amb_directed']),axis=0)
clip={'run_meander':subset}
plots.mean_bar_plus_conf(clip, ['Pre','Dur','Post'],
                         use_key='run_meander',ax=ax,
                         clip_method=False)
plt.ylim(0,3)
plt.ylabel('Directedness (cm/deg)')
subset=np.stack(list(data.loc[~chr2,'amb_bouts']),axis=0)
print(np.nanmean(subset,axis=0))


# %% Can I reasonably combine Ai32 & ChR2 mice?

# %% What percent of time ethovision says immobile is animal rearing?
print(sum(raw['ambulation'] & raw['dlc_is_rearing_logical']) / len(raw['im']) * 100)