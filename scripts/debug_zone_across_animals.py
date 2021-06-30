#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:40:23 2021

@author: brian
"""
from gittislab import signals, behavior, dataloc, ethovision_tools, plots, model
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import pdb
from itertools import compress
import scipy

ex0=['exclude','Bad','GPe','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','hm4di','Str','A2A','Ai32']]
# inc=[['AG','Str','D2_D1','ChR2_hM3Dq','10x10_15mW']]
make_preproc = False
exc=[ex0]
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
        
    basepath = 'F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\'
else:
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'


ex0=['exclude','Bad','GPe','bad','Broken', 'grooming',
     'Exclude','Other XLS']
exc=[ex0]
inc=[['AG','Str','A2A','Ai32','zone_1_0p5mw',]]
pns=dataloc.raw_csv(basepath,inc[0],ex0)

raw,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
# %% Plot zone day

if not isinstance(pns,list):
    pns=[pns]
saveit=True
closeit=False
for pn in pns:
    raw,meta=ethovision_tools.csv_load(pn,columns='All',method='preproc' )
    plots.plot_zone_day(raw,meta,save=saveit, close=closeit)

# %% Goal:
data = behavior.open_loop_summary_collect(basepath,inc,exc)
# %%
fig=plots.plot_zone_mouse_summary(data)
# %% Data collection debugging:


#Chunk mouse locations:
xx,yy=behavior.trial_part_position(raw,meta, chunk_method='task')
#Convert 2 2d histogram:
hist=[]
for x,y in zip(xx,yy):
    dat,xbin,ybin=np.histogram2d(x, y, bins=20, range=[[-25,25],[-25,25]], density=True)
    hist.append(dat)
# %%
data=behavior.experiment_summary_helper(raw,meta)
#%% 
data=behavior.zone_rtpp_summary_collect(basepath, inc,exc)
dat=np.stack(data['prob_density_arena'][use],axis=-1)  

# %%

# First a 1-D  Gaussian
t = np.linspace(-35, 35, 30)
bump = np.exp(-0.1*t**2)
bump /= np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

#%% For each animal, smooth the position estimate, rotate, then take mean
anids=np.unique(data['anid'])
dat=[]
for anid in anids: #Currently assume each row is a unique animal
    row = data.loc[:,'anid'].values == anid
    zone = int(data.loc[row,'proto'].values[0].split('_')[1])
    temp = data.loc[np.argwhere(row)[0][0],'prob_density_arena']
    
    for i,t in enumerate(temp):
        if zone == 2:
            t=np.flipud(t)
        # t = np.rot90(t,k=1)
        t = scipy.signal.fftconvolve(t, kernel, mode='same')
        temp[i]=t
    dat.append(temp)
dat=np.stack(dat,axis=-1)      
dat=np.mean(dat,axis=3)
# %% Debug plotting smoothed heatmaps of location:
use = 0
# dat=np.stack(data['prob_density_arena'][use],axis=-1)      


# img3 = scipy.signal.fftconvolve(img, kernel[:, :, np.newaxis], mode='same')
fig,f_row = plt.subplots(3,3)   
xx=data.loc[use,'x_task_position']
yy=data.loc[use,'y_task_position']
plots.pre_dur_post_arena_plotter(xx,yy,f_row[0,:])
for i,a in enumerate(f_row[1,:]):
    d=dat[i,:,:]
    # d=np.rot90(d)
    norm=colors.LogNorm(vmin=np.min(d[:])+0.0001, vmax=np.max(d[:]))
    a.pcolormesh(d,norm=norm, cmap='coolwarm',shading='auto')
    # pcm = a.pcolor(X, Y, Z,
    #                norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
    #                cmap='PuBu_r', shading='auto')
    # a.imshow(d,cmap='coolwarm',norm=norm,origin='lower',extent=[0,20,0,20])
for i,a in enumerate(f_row[2,:]):
    d=np.rot90(dat[i,:,:])
    norm=colors.LogNorm(vmin=np.min(d[:])+0.0001, vmax=np.max(d[:]))
    a.pcolormesh(d,norm=norm,cmap='coolwarm',shading='auto')
    
# %% Plot using plots:
plots.plot_zone_mouse_summary(data,example_mouse=1)