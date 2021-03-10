#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:04:16 2021

@author: brian
"""

import pandas as pd
import numpy as np
from pathlib import Path
from gittislab import dataloc, ethovision_tools, signal, plots, behavior, model
import matplotlib.pyplot as plt
import json 
import librosa 
basepath = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'


train_video_pn=[basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5363_2_BI121719/',
                basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_2_BI052819/', #File from low res camera
                basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5477_4_BI022620/',
                ]

train_video_boris_obs=['AG5363_2_BI121719',                       
                       'AG4486_2',
                       'AG5477_4',
                       ]

test_video_pn=[basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_3_BI060319/', #Low res, File that prev method struggled with (only 1 rear)
               basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/', # File from high res, Multiple rounds of refinement, file same as dlc_and_ka_rearing_confirmed_fa_final_v2.boris
               ]

test_video_boris_obs = ['AG4486_3', 'event_check']

use_cols =   ['vel','area', 'delta_area', 'elon', # 'dlc_front_over_rear_length'
              'dlc_side_head_x','dlc_side_head_y',
              'dlc_front_centroid_x','dlc_front_centroid_y',
              'dlc_rear_centroid_x','dlc_rear_centroid_y',
              'dlc_snout_x', 'dlc_snout_y',
              'dlc_top_head_x', 'dlc_top_head_y',
              'dlc_top_body_center_x', 'dlc_top_body_center_y',
              'dlc_top_tail_base_x','dlc_top_tail_base_y',
              'video_resolution','human_scored_rear','head_hind_px_height',
              'front_hind_px_height','side_length_px',
              'top_length_px']

# %% Test spectral data 


# %% Create training and test data from rear_scored videos:
train = model.combine_raw_csv_for_modeling(train_video_pn,train_video_boris_obs,
                                  use_cols)
train.to_csv('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/Train.csv')

test =  model.combine_raw_csv_for_modeling(test_video_pn,test_video_boris_obs,
                                 use_cols)
test.to_csv('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/Test.csv')

#%% NOTE: Currently the random forest model is in the fastai folder as a notebook on tabular data
# 'bi_tabular_rearing_model_experiments.ipynb'

# %% Test spectral data 




# %% Random Forest method predictions on validation ('Test.csv') set:
obs_path = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/obs_valid_rear.csv')
pred_path= Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/pred_valid_rear.csv')

# obs_path = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/obs_train_rear.csv')
# pred_path= Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/pred_train_rear.csv')

obs= pd.read_csv(obs_path)
pred= pd.read_csv(pred_path)
plt.figure()
plt.plot(obs['human_scored_rear'],'k')
plt.plot(pred['pred'],'r')
thresh=0.6
plt.plot([0,len(pred),],[thresh,thresh],'--b')
plt.plot((pred['pred'] > thresh)*1.25, '--b')

#%% Re-split validation set  
test_video_pn=[basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_3_BI060319/', #Low res, File that prev method struggled with (only 1 rear)
               basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/', # File from high res, Multiple rounds of refinement, file same as dlc_and_ka_rearing_confirmed_fa_final_v2.boris
               ]

use_cols =   ['time','vel','area', 'delta_area', 'elon', # 'dlc_front_over_rear_length'
              'dlc_side_head_x','dlc_side_head_y',
              'dlc_front_centroid_x','dlc_front_centroid_y',
              'dlc_rear_centroid_x','dlc_rear_centroid_y',
              'dlc_snout_x', 'dlc_snout_y',
              'dlc_top_head_x', 'dlc_top_head_y',
              'dlc_top_body_center_x', 'dlc_top_body_center_y',
              'dlc_top_tail_base_x','dlc_top_tail_base_y',
              'video_resolution','human_scored_rear','head_hind_px_height',
              'front_hind_px_height','side_length_px',
              'top_length_px']
test =  model.combine_raw_csv_for_modeling(test_video_pn,test_video_boris_obs,
                                 use_cols)
# %%

p=pred['pred'].values
ind = np.where(test['video_resolution'].values==704)
df1=test.iloc[ind[0],:]
df1.loc[:,'rf_rear_pred']=p[ind[0]]
ind = np.where(test['video_resolution'] == 1280)
df2=test.iloc[ind[0],:]
df2['rf_rear_pred']=p[ind[0]]
thresh=0.45
d=[df1, df2]
for df in d:
    fig = plt.figure()
    
    plt.plot(df['time'],df['human_scored_rear'],'k')
    plt.plot(df['time'],df['rf_rear_pred'],'r')
    plt.plot(df['time'],(df['rf_rear_pred']>thresh)*1.25,'b')
    mng = plt.get_current_fig_manager()
    mng.window.activateWindow()
    # mng.window.raise_()