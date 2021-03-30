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
import librosa.display
from sklearn import metrics

basepath = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'


train_video_pn=[basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5363_2_BI121719/',
                basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_2_BI052819/', #File from low res camera
                basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5477_4_BI022620/',
                basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_1_BI052819'] # Only 2-3 rears in this video

train_video_boris_obs=['AG5363_2_BI121719',                       
                       'AG4486_2',
                       'AG5477_4',
                       'AG4486_1']


test_video_pn=[basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_3_BI060319/', #Low res, File that prev method struggled with (only 1 rear)
               basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/', # File from high res, Multiple rounds of refinement, file same as dlc_and_ka_rearing_confirmed_fa_final_v2.boris
               basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4700_5_BI090519/'] #Add test

test_video_boris_obs = ['AG4486_3', 
                        'event_check',
                        'AG4700_5']

use_cols =   ['vel','area', 'delta_area', # 'dlc_front_over_rear_length'
              'dlc_side_head_x','dlc_side_head_y',
              'dlc_front_centroid_x','dlc_front_centroid_y',
              'dlc_rear_centroid_x','dlc_rear_centroid_y',
              'dlc_snout_x', 'dlc_snout_y',
              'dlc_side_left_fore_x','dlc_side_left_fore_y', 
              'dlc_side_right_fore_x', 'dlc_side_right_fore_y', 
              'dlc_side_left_hind_x', 'dlc_side_left_hind_y',
              'dlc_side_right_hind_x', 'dlc_side_right_hind_y',
              'dlc_top_head_x', 'dlc_top_head_y',
              'dlc_top_body_center_x', 'dlc_top_body_center_y',
              'dlc_top_tail_base_x','dlc_top_tail_base_y',
              'video_resolution','human_scored_rear',
              'side_length_px',
              'head_hind_5hz_pw',
              'snout_hind_px_height',
              'snout_hind_px_height_detrend',
              'front_hind_px_height_detrend',
              'side_length_px_detrend',] #'dlc_front_over_rear_length',


valid_video_pn = test_video_pn

test_video_boris_obs = test_video_boris_obs

# %% IF needed:
pn=dataloc.raw_csv(test_video_pn[0])
raw,meta = ethovision_tools.csv_load(pn,method='preproc')
raw,meta = ethovision_tools.add_dlc_helper(raw,meta,pn.parent)

# %% Create training and test data from rear_scored videos:
train = model.combine_raw_csv_for_modeling(train_video_pn,train_video_boris_obs,
                                  use_cols,rescale = True, 
                                  avg_model_detrend = True,
                                  z_score_x_y  = True,
                                  flip_y = True)

test =  model.combine_raw_csv_for_modeling(test_video_pn,test_video_boris_obs,
                                 use_cols,rescale = True,
                                 avg_model_detrend = True,
                                 z_score_x_y = True,
                                 flip_y = True)


plt.figure(),plt.plot(train['snout_hind_px_height'])
plt.figure(),plt.plot(train['dlc_snout_y'])


train.to_csv('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/Train_v3.csv')
test.to_csv('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/Test_v3.csv')
print('\n\nFinished')
#%% NOTE: Currently the random forest model is in the fastai folder as a notebook on tabular data
# 'bi_tabular_rearing_model_experiments.ipynb'

train = pd.read_csv('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/Train.csv')
test = pd.read_csv('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/Test.csv')

# %% Fit distance correction model for each experiment & average:
keep_dist_model=[]
keep_std_model=[]
keep_mean_model=[]
for pn in train_video_pn:
    plt.figure()
    rawpn=dataloc.raw_csv(pn)
    raw,meta = ethovision_tools.csv_load(rawpn,method='preproc')
    raw,meta = ethovision_tools.add_dlc_helper(raw,meta,pn)
    
    x = raw['y']
    x.fillna(method = 'ffill',inplace=True)
    x.fillna(method = 'bfill', inplace = True)
    
    sort_ind=np.argsort(x)
    x=x[sort_ind]
    
    y = raw['dlc_rear_centroid_y'] -raw['dlc_front_centroid_y']
    y.fillna(method='ffill',inplace=True)
    y.fillna(method='bfill',inplace=True)
    y=y[sort_ind]

    if meta['exp_room_number'][0]==228:
        y=y*1.455 #Scale pixels
    plt.title(pn)
    plt.plot(x,y,'.k')
    
    #First normalize based off distance from side camera:
    step=10
    xtemp=np.array([i for i in range(-30,30,step)]) + step / 2
    keep_x = []
    keep_std = []
    keep_mean = []
    for i, dist in enumerate(range(-30,35,step)):
        subx=(x>dist) & (x< (dist + step))        
        if any(subx):            
            mn = np.mean(y[subx])
            std= np.std(y[subx])
            y[subx]=y[subx]- mn            
            y[subx]=y[subx]/std
            keep_x.append(dist)
            keep_std.append(std)
            keep_mean.append(mn)
    
    
    p2=np.poly1d(np.polyfit(keep_x,keep_std,2))
    keep_std_model.append(p2)
    p3=np.poly1d(np.polyfit(keep_x,keep_mean,2))
    xs=[x for x in range(-30,30,1)]
    plt.plot(xs,p2(xs),'m')
    plt.plot(xs,p3(xs),'c')
    keep_mean_model.append(p3)
    
    #Next, detrend via fitting a line to this data
    p=np.poly1d(np.polyfit(x,y,1))
    y=y - p(x)

    plt.plot(xs,p(xs),'b')
    #Detrend y:
   
    keep_dist_model.append(p)


    plt.plot(x,y*30,'.r')

# %% Plot models
plt.figure()
use_x=range(-30,30,1)
for p in keep_dist_model:
    plt.plot(use_x,p(use_x))
    
avg_p=np.poly1d(np.mean(keep_dist_model,axis=0))  
plt.plot(use_x,avg_p(use_x),'--k')  
plt.figure()
for p in keep_std_model:
    plt.plot(use_x,p(use_x))
    
avg_p2=np.poly1d(np.mean(keep_std_model,axis=0))  
plt.plot(use_x,avg_p2(use_x),'--k')  

avg_mean_model=np.poly1d(np.mean(keep_mean_model,axis=0))
# %% Version using average models instead of exp. specific:
    
for pn in test_video_pn:
    plt.figure()
    rawpn=dataloc.raw_csv(pn)
    raw,meta = ethovision_tools.csv_load(rawpn,method='preproc')
    raw,meta = ethovision_tools.add_dlc_helper(raw,meta,pn)
    
    x = raw['y']
    x.fillna(method = 'ffill',inplace=True)
    x.fillna(method = 'bfill', inplace = True)
    
    sort_ind=np.argsort(x)
    x=x[sort_ind]
    
    # y = raw['dlc_rear_centroid_y'] -raw['dlc_front_centroid_y'] #Trained on this
    # y = raw['dlc_rear_centroid_y'] -raw['dlc_snout_y'] #Works
    y= signal.calculateDistance(raw['dlc_front_centroid_x'].values,
                                raw['dlc_front_centroid_y'].values,
                                raw['dlc_rear_centroid_x'].values,
                                raw['dlc_rear_centroid_y'].values)
    y=pd.core.series.Series(y)
    y.fillna(method='ffill',inplace=True)
    y.fillna(method='bfill',inplace=True)
    y=y[sort_ind]

    if meta['exp_room_number'][0]==228:
        y=y*1.455 #Scale pixels
    plt.title(pn)
    ytemp=y
    
    y=y-avg_mean_model(x)
    y=y/avg_p2(x) # Predicted std. as opposed to actual

    plt.plot(xs,avg_p(xs),'b')
    #Detrend y:
   
    plt.plot(x,y*25,'.r')
    plt.plot(x,ytemp,'.k')
# %% Test spectral data as indicator of grooming / not rearing

dat = test['head_hind_px_height']
dat.fillna(method='ffill',inplace=True)
Fs = 29.97
n_fft = 256
hop_length=10
freqs = np.arange(0, 1 + n_fft / 2) * Fs / n_fft
S = librosa.feature.melspectrogram(y=dat.values, sr=Fs, n_fft= n_fft, hop_length=hop_length)
# im.show(spec)
plt.figure()
librosa.display.specshow(librosa.power_to_db(S, ref=np.max),sr=Fs,hop_length=hop_length
                         , y_axis='mel', fmax=15, x_axis='time')
# %% 
# plt.figure(),plt.plot(freqs[0:-1],np.log10(S[:,16]))

#Putative "grooming" band:
dm = signal.get_spectral_band_power(dat,29.97,4.5,6.5)
total_dur= dat.shape[0] / Fs
# time_s= np.linspace(0,total_dur,S.shape[1])
time_s= np.linspace(0,total_dur,dat.shape[0])

plt.figure(),
plt.plot(time_s,np.log10(dm) * 10 )


# plt.plot(time_s, np.log10(dm2))

plt.plot(time_s,test['head_hind_px_height'])
plt.plot(time_s,pred['pred']*30,'r')
plt.plot(time_s,obs['human_scored_rear']*40,'k')

# %% Load and examine ensemble predictions on the training set itself:
    


# %% Training set ROC, determine threshold for P = 0.05:
# obs_path = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/obs_train_rearv3.csv')
# pred_path= Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/nn_train_pred_rear_v3.csv')
obs_path = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/obs_train_rearv3.csv')
pred_ens_path= Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/nn_rf_ens_pred_train_rear_v3.csv')

obs= pd.read_csv(obs_path)
pred= pd.read_csv(pred_ens_path)
use_pred =pred['pred'].values
#See if smoothing / low-pass filtering improves:
use_pred =  signal.pad_lowpass_unpad(use_pred,1,29.97)

targets =obs['human_scored_rear'].values.astype(np.int16)
fpr,tpr,thr =metrics.roc_curve(targets,use_pred)
plt.figure()
plt.plot(fpr,tpr,'k')
plt.xlabel('False Alarm Rate')
plt.ylabel('Hit Rate')
fr_rate_desired = 0.05
ok_thresh = thr[fpr<fr_rate_desired][-1]
hit_at_thr=tpr[fpr<fr_rate_desired][-1]
plt.plot([fr_rate_desired,fr_rate_desired],[0,1],'--k')
plt.title('Thresh at %1.3f, hit rate = %1.3f, fa rate = %1.3f' % (ok_thresh,hit_at_thr,fr_rate_desired))

#Optimal threshold using Youlden's J-statistic
ix=np.argmax(tpr-fpr)
print('Youlden J optimal thresh hit = %1.3f, fa = %1.3f, thresh = %.3f'\
      % (tpr[ix],fpr[ix],thr[ix]))

# %% ROC performance on validation set:
obs_path = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/obs_valid_rearv3.csv')
# pred_path= Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/nn_pred_valid_rear_v3.csv')
pred_ens_path = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/nn_rf_ens_pred_valid_rear_v3.csv')
obs= pd.read_csv(obs_path)
pred= pd.read_csv(pred_ens_path)
use_pred =pred['pred'].values
#See if smoothing / low-pass filtering improves:
use_pred =  signal.pad_lowpass_unpad(use_pred,1,29.97)

targets =obs['human_scored_rear'].values.astype(np.int16)
fpr,tpr,thr =metrics.roc_curve(targets,use_pred)
ix=np.argmax(tpr-fpr)
print('Youlden J optimal thresh hit = %1.3f, fa = %1.3f, thresh = %.3f'\
      % (tpr[ix],fpr[ix],thr[ix]))
plt.figure()
plt.plot(fpr,tpr,'k')
plt.xlabel('False Alarm Rate')
plt.ylabel('Hit Rate')
# fr_rate_desired = 0.05
ok_thresh = 0.296
fr_rate_given_train_thresh = fpr[thr>ok_thresh][-1]
hit_at_thr=tpr[thr>ok_thresh][-1]
plt.plot([fr_rate_given_train_thresh ,fr_rate_given_train_thresh ],[0,1],'--r')
plt.title('Thresh at %1.3f, hit rate = %1.3f, fa rate = %1.3f' % (ok_thresh,hit_at_thr,fr_rate_given_train_thresh))


# %% Perform above process with data I have never looked at before:

# ffn = basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4700_5_BI090519/'
# boris_obs = 'AG4700_5'

#WARNING: Human scored times are currently delayed / offset:
# ffn = basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4109_2_BI060519/'
# boris_obs = 'AG4109_2'

ffn = basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_3_BI060319/'
boris_obs = 'AG4486_3'

# ffn = basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5362_3_BI022520/'
# boris_obs = 'event_check'

#WARNING: Human scored times are currently delayed /offset:
# ffn = basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5769_1_BI022520' 
# boris_obs='AG5769_1'

# ffn = basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_1_BI052819' #Very few actual rears in this video
# boris_obs='AG4486_1'
rf_model_fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/rf_model_v3.joblib'
weights_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/bi_rearing_nn_weightsv3'
tab_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/to_nnv3.pkl'
auroc,pred,human_scored = model.rear_nn_auroc_perf(ffn,boris_obs,
                                                       prob_thresh = 0.295,
                                                       low_pass_freq=1,
                                                      tab_fn = tab_fn,
                                                      weights_fn = weights_fn,
                                                      rf_model_fn = rf_model_fn,
                                                      )
plt.figure()
plt.plot(pred,'r')
plt.plot(human_scored,'k')
plt.title('%s AUROC: %1.4f' % (boris_obs,auroc))

# %% Calculate average AUROC for training set:

ka=[]
kp=[]
kh=[]

for ffn,boris_obs in zip(test_video_pn,test_video_boris_obs):
    rf_model_fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/rf_model_v3.joblib'
    weights_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/bi_rearing_nn_weightsv3'
    tab_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/to_nnv3.pkl'
    auroc,pred,human_scored = model.rear_nn_auroc_perf(ffn,boris_obs,
                                                       prob_thresh = 0.296,
                                                       low_pass_freq=1,
                                                      tab_fn = tab_fn,
                                                      weights_fn = weights_fn,
                                                      rf_model_fn = rf_model_fn,
                                                      )
    ka.append(auroc)
    kp.append(pred)
    kh.append(human_scored)
    
print(np.mean(ka))