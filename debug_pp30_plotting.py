#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:04:21 2020

@author: brian
"""

from gittislab import signal, behavior, dataloc, ethovision_tools, plots
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pdb
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
# %% Fix bugs 1 by 1..... and repeat
# Make .h5
# inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim',]]
inc = [['AG','GPe','CAG','Arch','Bilateral','AG4700_5','pp30_cond_dish_fc_stim']] # 'zone_1_30mW'
exc = [['exclude','Str','_and_SNr','_and_Str','Right','grooming',]]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
xlsx_paths=dataloc.rawxlsx(basepath,inc[0],exc[0])
# df=pd.read_excel(xlsx_paths[1],sheet_name=None,na_values='-',header=None) #Key addition
raw,params=ethovision_tools.raw_params_from_xlsx(xlsx_paths[1])

# v=behavior.smooth_vel(raw,params,win=10)
# %%
tt=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/pp30_cond_dish/AG4486_1_KA061719') #File can not currently be read back in
test_file=dataloc.rawxlsx(tt)
raw,params=ethovision_tools.raw_params_from_xlsx(test_file)
# %% Test save as h5
pn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/pp30_cond_dish_fc_stim/AG4486_3_KA062119/Raw_AG4486_3_KA062119.h5')
ethovision_tools.h5_store(pn,raw)
# %% Load
store = pd.HDFStore(pn)
data = store['mydata']
# %% Test on all cag arch bilateral trials:
# inc = [['AG','GPe','CAG','Arch']] # 'zone_1_30mW'
inc = [['AG','A2A','ChR2','Str']] # 'zone_1_30mW'
exc = [['exclude','_and_SNr','_and_Str','20min_10Hz',
        'grooming','20min_4Hz','Exclude','Other XLS']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_to_csv(basepath,inc,exc,force_replace=False)

# %% Print summary of metadata retrieved by query:
# inc = [['AG','A2a','ChR2','Str']] # 'zone_1_30mW'
# exc = [['exclude','_and_SNr','_and_Str','20min_10Hz',
#         'grooming','20min_4Hz','Exclude']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary)
plt.hist(summary.stim_n)
# %% Load example data
inc=[['AG','GPe','CAG','Arch','10x30']]
exc=[['exclude','_and_Str','Left','Right','Other XLS','Exclude']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
raw,meta=ethovision_tools.csv_load(pns[1])

# %% Attempt to chunk velocity by stim start times:
useday='stim'
if useday == 'nostim':
    #Conditioning days with no stim:
    inc=[['AG','GPe','CAG','Arch','pp30_cond_dish']] # 
    exc=[['exclude','_and_Str','Left','Right','pp30_cond_dish_fc_stim']]
else:
    #Days with stim:
    inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim']] # 
    exc=[['exclude','_and_Str','Left','Right']]

basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)   
for i,pn in enumerate(pns):
    fig,ax = plt.subplots(figsize=(15, 5))

    raw,meta=ethovision_tools.csv_load(pn)

    # chunk(x,y,x_points,x_range)
    x=raw['time'].values
    y=raw['vel'].values
    x_points=meta['stim_on'].values
    x_range=[-1,3] #Seconds
    
    output=signal.chunk_by_x(x,y,x_points,x_range)
    vel_clips=np.array(output)
    x_val=x[0:vel_clips.shape[1]]+x_range[0]
    
    # Chunk trajectories and ultimately color by velocity
    y1=raw['x'].values
    y2=raw['y'].values
    output=signal.chunk_by_x(x,y1,x_points,x_range)
    x_clips=np.array(output)
    x_val=x[0:x_clips.shape[1]]+x_range[0]
    
    output=signal.chunk_by_x(x,y2,x_points,x_range)
    y_clips=np.array(output)
    #plt.plot(x_clips.T,y_clips.T,'k',alpha=0.01)
    plt.scatter(x_clips,y_clips, s=vel_clips*3, c=vel_clips,alpha=0.1)
    plt.title('%s with %s' % (summary['anid'][i],summary['settings'][i]))
    
# %%
'''
OK from the above analysis it's pretty clear that pp30_cond_dish are not run 
with the same level of zone detection or stim parameters as pp30_cond_dish_fc_stim
... which matters in so much as comparing corner behavior will be difficult.
OR, break it down into pre- during- and post- periods
'''
# %% Determine how zones map onto 'front_dot' and 'back_dot'
#Front dot zone stim = 'In zone 4'

raw,meta=ethovision_tools.csv_load(pns[-1])
corners=['In zone 2','In zone 3','In zone 4','In zone 5']
overlap=[]
for corner in corners:
    overlap.append(np.nansum(raw[corner].astype('bool') & raw['laserOn'].astype('bool')))
fig,ax=plt.subplots(1,1)
plt.bar(corners,overlap)
plt.title(meta.loc[0,'etho_trial_control_settings'])

#Back dot zone stim = 'In zone 5'
raw,meta=ethovision_tools.csv_load(pns[0])
corners=['In zone 2','In zone 3','In zone 4','In zone 5']
overlap=[]
for corner in corners:
    overlap.append(np.nansum(raw[corner].astype('bool') & raw['laserOn'].astype('bool')))
fig,ax=plt.subplots(1,1)
plt.bar(corners,overlap)
plt.title(meta.loc[0,'etho_trial_control_settings'])

# %% Normalize coordinates for each recording:
useday='stim'
exclude_exp=['AG4486_1_KA062019']
if useday == 'nostim':
    #Conditioning days with no stim:
    inc=[['AG','GPe','CAG','Arch','pp30_cond_dish']] # 
    exc=[['exclude','_and_Str','Left','Right','pp30_cond_dish_fc_stim']]
else:
    #Days with stim:
    inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim']] # 
    exc=[['exclude','_and_Str','Left','Right','AG4486_1_KA062019']]
    
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.raw_csv(basepath,inc[0],exc[0])
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)  
stim_x_max=[]
stim_y_max=[]
for i,pn in enumerate(pns):
    fig,axes = plt.subplots(figsize=(15, 5))

    raw,meta=ethovision_tools.csv_load(pn)
    stim_type=meta.loc[0,'etho_trial_control_settings']
    anid=meta.loc[0,'etho_animal_id']
    xn,yn=behavior.norm_position(raw) #Center x,y coordinates of mouse on 0,0
    
    
    #Identify pre, during, and post. (there is not pre during and post)
    #plot all movement:
    
    
    #If stim day, look at some other info:
    if useday == 'stim':
        if stim_type == 'front_dot_zone_stim':
            yn = -1*yn #flip up/down
            stim_corner='In zone 4'
        elif stim_type == 'back_dot_zone_stim':
            stim_corner='In zone 5'
        plt.plot(xn,yn,'.k')
        
        #plot stimulated corner in red:
        in_stim=raw[stim_corner].astype('bool')
        plt.plot(xn[in_stim],yn[in_stim],'ro')
        plt.title('%s %s' % (anid,stim_type))
    
        #Collect the boundaries of stimulated region:
        stim_x_max.append(np.nanmax(xn[in_stim]))
        stim_y_max.append(np.nanmax(yn[in_stim]))
    else:
        plt.plot(xn,yn,'.k')
# %%
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)   
fd=summary['settings'].values=='front_dot_zone_stim'
plt.subplots(1,1)
plt.plot(stim_x_max,stim_y_max,'.k') 
plt.plot(np.array(stim_x_max)[fd],np.array(stim_y_max)[fd],'.r') 

# %% Is there ever pre, during, post (?) --> no.
early=[]
for i,pn in enumerate(pns):
    raw,meta=ethovision_tools.csv_load(pn)
    early.append(np.min(meta['stim_on']) < 60*2) # There is always a stim in under 2 minutes
print(early)

# %% For each mouse, plot conditioning days on left, stim days on right
mice=['AG4700_5','AG4486_1','AG4486_3']
usedays=['stim','nostim']
stimx=-14.3 #Normalized max x boundary of stim area (for no-stim days)
stimy=-0.23 #Normalized max y boundary of stim area (for no-stim days)
xedges=np.linspace(-25,25,101)
yedges=np.linspace(-11,11,45)
plt.close('all')
dx = (xedges[1]-xedges[0])/2.
dy = (yedges[1]-yedges[0])/2.
extent = [xedges[0]-dx, xedges[-1]+dx, yedges[0]-dy, yedges[-1]+dy]
for mouse in mice:
    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(15, 5))
    for j,useday in enumerate(usedays):
        if useday == 'nostim':
            #Conditioning days with no stim:
            inc=[['AG','GPe','CAG','Arch','pp30_cond_dish',mouse]] # 
            exc=[['exclude','_and_Str','Left','Right','pp30_cond_dish_fc_stim']]
        else:
            #Days with stim:
            inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim',mouse]] # 
            exc=[['exclude','_and_Str','Left','Right','AG4486_1_KA062019']]
            
        basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
        pns=dataloc.raw_csv(basepath,inc[0],exc[0])
        summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)  
        
        #Better way: create 2D histogram
        keep_hist=np.zeros((100,44))
        # fig,axes = plt.subplots(figsize=(15, 5))
        for i,pn in enumerate(pns):               
            raw,meta=ethovision_tools.csv_load(pn)
            stim_type=meta.loc[0,'etho_trial_control_settings']
            anid=meta.loc[0,'etho_animal_id']
            xn,yn=behavior.norm_position(raw) #Center x,y coordinates of mouse on 0,0
            ax=0
            if useday == 'stim': #'stim' should always come first, so this can be set correctly for 'nostim' day
                ax=1
                if stim_type == 'front_dot_zone_stim':
                    flip_y=True
                    stim_corner='In zone 4'
                elif stim_type == 'back_dot_zone_stim':                    
                    stim_corner='In zone 5'

            if flip_y == True:
                yn = -1*yn #flip up/down
           
            hist, _, _ = np.histogram2d(xn, yn, (xedges, yedges))
            hist=hist / np.nansum(hist.flat)
            keep_hist=keep_hist+hist
            
        hist_norm=np.log10(((100*keep_hist)/(i+1))+1)
        hist_norm[hist_norm==-np.inf]=0
        hist_norm=hist_norm/np.max(hist_norm)
        axes[ax].imshow(hist_norm.transpose(),
                    vmax=0.05,
                    interpolation='none',
                    cmap='binary',
                    extent=extent)
        axes[ax].add_patch(Rectangle((-24,abs(stimy)),
                                     width=25+stimx,
                                     height=11+stimy,
                                     fill=False,
                                     color='r',
                                     lw=3))
        axes[ax].set_title('%s %s' % (anid,useday))
        axes[ax].set_xlabel('cm')
        axes[ax].set_ylabel('cm')