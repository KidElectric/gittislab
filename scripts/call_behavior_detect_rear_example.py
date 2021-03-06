#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:13:18 2020

@author: brian
"""
from matplotlib import pyplot as plt
#from scipy.ndimage import gaussian_filter
import sys
sys.path.append('/home/brian/Dropbox/Python')
from gittislab import dataloc
from gittislab import behavior
from gittislab import mat_file
from gittislab import plots
import time
import numpy as np
import pandas as pd
from scipy import stats
import cv2

# %% Test how long it takes to run a rear-extraction frunction on 1 video:
inc=['GPe','Arch','PV','10x30','Bilateral','AG3474_1'] #1st test
# inc=['10x30','A2A','AG3525_10','Bilateral']
exc=['exclude']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior'
paths=dataloc.gen_paths_recurse(basepath,inc,exc,'.h5')

vid_path=dataloc.video(basepath,inc,exc)
print('Finished')
save_figs=False
tic = time.perf_counter()
peak,start,stop,df = behavior.detect_rear(paths,rear_thresh=0.7,min_thresh=0.2,save_figs=save_figs,
                dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1)
toc = time.perf_counter()
print('%2.1f seconds.' % float(toc-tic))

# %% Script that opens df, checks if fields are present, if not, adds it, saves .h5
# similarly
inc=['GPe','Arch','PV','10x30','Bilateral','AG3474_1'] #1st test
exc=['SNr','exclude']
# paths=dataloc.gen_paths_recurse(basepath,inc,['exclude'],'.xlsx')
paths=dataloc.rawxlsx(basepath,inc,exc)
paths=dataloc.rawmat(basepath,inc,exc)
for path in paths:
    ns=''
    for i,p in enumerate(path.parts):
        if i > 5 or i == len(path.parts):
            ns=ns+'-'+path.parts[i]
    print(ns)
tic = time.perf_counter()
xl=pd.read_excel(paths[0],sheet_name=0,na_values='-',header=38)
toc = time.perf_counter()
print('%2.1f seconds.' % float(toc-tic))
if xl['Trial time'][0] =='(s)':
    xl=xl.drop([0],axis=0)
    
# %% Compare rearing vs. light stim in one or a few conditions
conds_inc=[['Str','D1','ChR2','10x'],
           ['Str','A2A','ChR2','10x30'],
           ['Str','D1','Arch','10x10_30mW']]
conds_exc=[['Caspase','exclude','hSyn','EYFP','D2_D1'],
           ['Caspase','exclude','hSyn','EYFP','D2_D1'],
           ['Caspase','exclude','hSyn','EYFP','D2_D1']]
labels=['Str-D1-ChR2','Str-D2-ChR2','Str-D1-Arch']
use_move=True
# out = behavior.prob_rear_stim_dict(basepath,conds_inc,conds_exc,labels,use_move)
ylab='P(Rearing)'
out = behavior.rear_rate_stim_dict(basepath,conds_inc,conds_exc,labels,use_move)
ylab='Rears / s'
# %%
fig,ax=plt.subplots(2,2,figsize=(11,5))
plot_order=labels
normalized = False
if normalized==True:
    ymax=2.5
else:
    ymax=0.5
for axi,key in enumerate(plot_order):
    dat=out[key]
    i=np.unravel_index(axi,ax.shape,'F')
    for row in dat:
        x=[1,2]
        y=row
        if normalized==True:
            y=y/row[0]
        ax[i].plot(x,y,'-ok')
    ax[i].xaxis.set_ticks(x)
    t,p=stats.ttest_rel(dat[:,0],dat[:,1])
    plots.sig_star(x,p,ax[i])
    # if (axi+1) % (ax.shape[0]) == 0:
    ax[i].xaxis.set_ticklabels(['Off','On'],rotation=0)
    # else:
    #     ax[i].xaxis.set_ticklabels(['',''])
    
    ax[i].set_title(key)
    ax[i].set_xlim([0,3])
    ax[i].set_ylim([0,ymax])
    if (axi+1) / ax.shape[0] <= 1:
        ax[i].set_ylabel(ylab)
    else:
        ax[i].tick_params(axis='y',label1On=False)

plots.cleanup(ax)
mngr = plt.get_current_fig_manager()
# %% Compare rearing vs. light stimulation in many conditions:
    
basepath,conds_inc,conds_exc,labels=dataloc.common_paths()

#If desired: only rerun a subset:
# conds_inc=conds_inc[-1:]
# conds_exc=conds_exc[-1:]
# labels=labels[-1:]

use_move=True
tic = time.perf_counter()
out = behavior.rear_rate_stim_dict(basepath,conds_inc,conds_exc,labels,use_move)
ylab='Rears / s'
# Save 'out'
savefn='~/Dropbox/Gittis Lab Data/Brian/DLC_Analysis/processed_output/rear_rate_laseroff_v_on.pickle'

# out = behavior.prob_rear_stim_dict(basepath,conds_inc,conds_exc,labels,use_move)
# ylab='P(Rearing)'
# savefn='~/Dropbox/Gittis Lab Data/Brian/DLC_Analysis/processed_output/rear_probability_laseroff_v_on.pickle'

toc = time.perf_counter()
print('Full analysis took: %2.1f seconds.' % float(toc-tic))

pd.to_pickle(out,savefn)
print('File saved to %s' % savefn)

# %% Load 'out' from above analysis:
savefn='~/Dropbox/Gittis Lab Data/Brian/DLC_Analysis/processed_output/rear_probability_laseroff_v_on.pickle'
out=pd.read_pickle(savefn)

# %%
savefn='~/Dropbox/Gittis Lab Data/Brian/DLC_Analysis/processed_output/rear_rate_laseroff_v_on.pickle'
out=pd.read_pickle(savefn)
# %% Plot probability of rearing under all conditions:
    
fig,ax=plt.subplots(2,6,figsize=(11,5))
reorder=[1, 5, 11, 10, 0, 4, 2, 3, 6, 7, 8, 9] # [labels.index(lab) for lab in plot_order]
plot_order=[labels[i] for i in reorder]
normalized = False
if normalized==True:
    ymax=2.5
else:
    ymax=0.5
for axi,key in enumerate(plot_order):
    dat=out[key]
    i=np.unravel_index(axi,ax.shape,'F')
    for row in dat:
        x=[1,2]
        y=row
        if normalized==True:
            y=y/row[0]
        ax[i].plot(x,y,'-ok')
    ax[i].xaxis.set_ticks(x)
    t,p=stats.ttest_rel(dat[:,0],dat[:,1])
    plots.sig_star(x,p,ax[i])
    if (axi+1) % (ax.shape[0]) == 0:
        ax[i].xaxis.set_ticklabels(['Off','On'],rotation=0)
    else:
        ax[i].xaxis.set_ticklabels(['',''])
    
    ax[i].set_title(key)
    ax[i].set_xlim([0,3])
    ax[i].set_ylim([0,ymax])
    if (axi+1) / ax.shape[0] <= 1:
        ax[i].set_ylabel(ylab)
    else:
        ax[i].tick_params(axis='y',label1On=False)

plots.cleanup(ax)
mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
mngr.window.setGeometry(500,2000,1400,600) #Dist from left, dist from top, width, height




# %% Find higher resolution videos to check rear detection:
basepath,conds_inc,conds_exc,labels=dataloc.common_paths()
larger_vids=[]
smaller_vids=[]
for i,inc in enumerate(conds_inc):
    exc=conds_exc[i]
    vid_paths=dataloc.video(basepath,inc,exc)
    for path in vid_paths:
        cap=cv2.VideoCapture(str(path))
        if (cap.isOpened()) == True:
            width  = cap.get(3) # float
            height = cap.get(4) # float
            if (width > 1000):
                print('%s has dims %d x %d' % (str(path),width,height))
                h5_path=dataloc.gen_paths_recurse(path.parent,filetype='.h5')
                # if any(h5_path):
                larger_vids.append(h5_path)
            else:
                print('%s has dims %d x %d' % (str(path),width,height))
                h5_path=dataloc.gen_paths_recurse(path.parent,filetype='.h5')
                # if any(h5_path):
                smaller_vids.append(h5_path)
        else:
            print('%s not opened' % str(path))
# %% Check rears in one large video:
save_figs=True
tic = time.perf_counter()
peak,start,stop,df = behavior.detect_rear(larger_vids[-1],rear_thresh=0.7,min_thresh=0.2,save_figs=save_figs,
                dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1)
#Note: /home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/A2A/Ai32/Bilateral/10x10/AG5363_2_BI121719/Trial    38.mpg is also B&W -- looks good

# %% Test a generalized frame extraction function:
framesets=np.concatenate((start[...,None],peak[...,None],stop[...,None]),axis=1)
framesets=framesets[0:3,:]
potential_parts=np.unique([col[1] for col in df.columns])
parts=['front_centroid','rear_centroid','head_centroid']
plot_points=plots.gen_sidecam_plot_points(df,parts,framesets);

    
cap=plots.etho_check_sidecamera(dataloc.video(larger_vids[-1].parent),framesets,plot_points)











