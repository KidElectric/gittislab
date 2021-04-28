#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:16:15 2021

@author: brian
"""
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
from pathlib import Path
from gittislab import dataloc, ethovision_tools, signals, plots, behavior
import matplotlib.pyplot as plt
import json 
import pdb
import fastai.tabular.all as fasttab
from sklearn import metrics
import joblib
from scipy.optimize import curve_fit

def fit_sigmoid(xdata,ydata,method='dogbox'):
    p0 = [max(ydata), np.median(xdata),1,min(ydata)] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method=method)
    return popt,pcov

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)

def combine_raw_csv_for_modeling(raw_pns,
                                 boris_obs_names,
                                 use_cols,
                                 uniform_boris_fn='Rearing Observations.boris',
                                 rescale = False,
                                 avg_model_detrend = False,
                                 z_score_x_y = False,
                                 flip_y = False,
                                 meta_to_raw = []):
    
    combined = pd.DataFrame([],columns=use_cols)

    for pn,obs in zip(raw_pns,boris_obs_names):
        p=Path(pn)
        inc=[['AG']]
        exc=[['exclude']]
        ethovision_tools.unify_raw_to_csv(p,inc,exc)
        ethovision_tools.raw_csv_to_preprocessed_csv(p,inc,exc,force_replace=False)
        pns=dataloc.raw_csv(p,inc[0],exc[0])
        raw,meta=ethovision_tools.csv_load(pns,method='raw')
        
        fn_ka = p.joinpath(uniform_boris_fn)
        f = open(fn_ka,'r')
        boris =  json.loads(f.read())
        # Get human scored rearing events and add as vector!
        human_scored_rearing = behavior.boris_to_logical_vector(raw,boris,obs,'a','d')
        
        dlc=ethovision_tools.add_dlc_helper(raw,meta,p,force_replace = True)
        dlc=dlc[0]     
        # pdb.set_trace()
        if meta['exp_room_number'][0] == 228:
            x_scale = 512/480
            y_scale = 1.455 #Scale pixels #Video takes up only half of screen in these recordings
            vid_res= np.ones((dlc.shape[0],1))  * 704 #Video width 704 x 480
            # pdb.set_trace()
        else:
            x_scale = 1
            y_scale = 1
            vid_res = np.ones((dlc.shape[0],1)) * 1280 #Video width   height = 512                         
            
        
        if rescale==True:
            for col in dlc.columns:
                if ('dlc' in col) and ('x' in col):
                    dlc[col]=dlc[col].values * x_scale
                    
                if ('dlc' in col) and ('y' in col):                    
                    dlc[col]=dlc[col].values * y_scale 
                             
        dlc['video_resolution'] = vid_res
        dlc['human_scored_rear'] = human_scored_rearing
        # dlc['head_hind_px_height']= dlc ['dlc_rear_centroid_y'] - dlc['dlc_side_head_y']

        dlc['front_hind_px_height'] = dlc ['dlc_rear_centroid_y'] - dlc['dlc_front_centroid_y']
        dlc['head_hind_5hz_pw'] = signals.get_spectral_band_power(dlc['front_hind_px_height'],
                                                                 meta['fs'][0],4.5,6.5)
        dlc['snout_hind_px_height'] = dlc ['dlc_rear_centroid_y'] - dlc['dlc_snout_y']
        dlc['side_length_px']=signals.calculateDistance(dlc['dlc_front_centroid_x'].values,
                                                       dlc['dlc_front_centroid_y'].values,
                                                       dlc ['dlc_rear_centroid_x'].values,
                                                       dlc ['dlc_rear_centroid_y'].values)
        # dlc['top_length_px']=signals.calculateDistance(dlc['dlc_top_head_x'].values,
        #                                                dlc['dlc_top_head_y'].values,
        #                                                dlc ['dlc_top_tail_base_x'].values,
        #                                                dlc ['dlc_top_tail_base_y'].values)
        #Detrend effect of mouse distance from camera using an average pre-fitted
        #z-score approach:
        if avg_model_detrend == True:
            detrend_cols=['snout_hind_px_height',
                          'front_hind_px_height',
                          'side_length_px']
            for col in detrend_cols:
                y=dlc[col]
                x=dlc['x']
                dlc[col+'_detrend'] = average_z_score(x,y)
        
        if z_score_x_y == True:
            for col in dlc.columns:
                if ('dlc' in col) and (('x' in col) or ('y' in col)):
                    temp=dlc[col].values 
                    temp = temp - np.nanmean(temp)
                    temp = temp / np.nanstd(temp)
                    dlc[col] = temp

        if flip_y == True:
            for col in dlc.columns:
                if ('dlc' in col) and  ('y' in col):
                    dlc[col] = -1 * dlc[col]
                    # dlc[col] = dlc[col] - np.nanmin(dlc[col])
                    
        for col in meta_to_raw:
            dlc[col]=meta[col][0]
            
        temp=dlc[use_cols]    
        combined = pd.concat([combined,temp])
        
    #Add time lags (?)
    combined.reset_index(drop=True, inplace=True)
    return combined

# def combine_raw_csv_for_predicting(raw_pns,
#                                  use_cols,
#                                  rescale = False,
#                                  avg_model_detrend = False,
#                                  z_score_x_y = False,
#                                  flip_y = False,
#                                  meta_to_raw = []):
    
#     combined = pd.DataFrame([],columns=use_cols)

#     for pn in raw_pns:
#         p=Path(pn)
#         inc=[['AG']]
#         exc=[['exclude']]
#         # ethovision_tools.unify_raw_to_csv(p,inc,exc)
#         ethovision_tools.raw_csv_to_preprocessed_csv(p,inc,exc,force_replace=False)
#         pns=dataloc.raw_csv(p,inc[0],exc[0])
#         raw,meta=ethovision_tools.csv_load(pns,method='raw')
#         raw = raw_to_prediction_format(raw,meta)
#         temp=raw[use_cols]    
#         combined = pd.concat([combined,temp])
        
#     #Add time lags (?)
#     combined.reset_index(drop=True, inplace=True)
#     return combined

def raw_to_prediction_format(raw,meta,
                             rescale=False,
                             avg_model_detrend=False,
                             z_score_x_y = False,
                             flip_y = False,
                             meta_to_raw = []): 
    
    # ethovision_tools.unify_raw_to_csv(p,inc,exc)
    # ethovision_tools.raw_csv_to_preprocessed_csv(p,inc,exc,force_replace=False)
    # pns=dataloc.raw_csv(p,inc[0],exc[0])
    # raw,meta=ethovision_tools.csv_load(pns,method='raw')
    
    if meta['exp_room_number'][0] == 228:
        x_scale = 512/480
        y_scale = 1.455 #Scale pixels #Video takes up only half of screen in these recordings
        vid_res= np.ones((raw.shape[0],1))  * 704 #Video width 704 x 480
        # pdb.set_trace()
    else:
        x_scale = 1
        y_scale = 1
        vid_res = np.ones((raw.shape[0],1)) * 1280 #Video width   height = 512                         
        
    if rescale==True:
        for col in raw.columns:
            if ('dlc' in col) and ('x' in col):
                raw[col]=raw[col].values * x_scale
                
            if ('dlc' in col) and ('y' in col):                    
                raw[col]=raw[col].values * y_scale 
                         
    raw['video_resolution'] = vid_res
    raw['human_scored_rear'] = False
    
    raw['front_hind_px_height'] = raw['dlc_rear_centroid_y'] - raw['dlc_front_centroid_y']
    raw['head_hind_5hz_pw'] = signals.get_spectral_band_power(raw['front_hind_px_height'],
                                                             meta['fs'][0],4.5,6.5)
    raw['snout_hind_px_height'] = raw['dlc_rear_centroid_y'] - raw['dlc_snout_y']
    raw['side_length_px']=signals.calculateDistance(raw['dlc_front_centroid_x'].values,
                                                   raw['dlc_front_centroid_y'].values,
                                                   raw ['dlc_rear_centroid_x'].values,
                                                   raw ['dlc_rear_centroid_y'].values)
    
    #z-score approach:
    if avg_model_detrend == True:
        detrend_cols=['snout_hind_px_height',
                      'front_hind_px_height',
                      'side_length_px']
        for col in detrend_cols:
            y=raw[col]
            x=raw['x']
            raw[col+'_detrend'] = average_z_score(x,y)
    
    if z_score_x_y == True:
        for col in raw.columns:
            if ('dlc' in col) and (('x' in col) or ('y' in col)):
                temp=raw[col].values 
                temp = temp - np.nanmean(temp)
                temp = temp / np.nanstd(temp)
                raw[col] = temp
    
    if flip_y == True:
        for col in raw.columns:
            if ('dlc' in col) and  ('y' in col):
                raw[col] = -1 * raw[col]
                # raw[col] = raw[col] - np.nanmin(raw[col])
                
    for col in meta_to_raw:
        raw[col]=meta[col][0]
        
    return raw

def average_z_score(x,y,avg_mean_model = None,avg_std_model = None):
    #Apply a z-score to y as a function of x from a relationship fitted from multiple distributions

    y=pd.core.series.Series(y)
    y.fillna(method='ffill',inplace=True)
    y.fillna(method='bfill',inplace=True)
    # Default example models were fit on the following videos (see: debug_prep_train_set_rearing_model.py):
    # train_video_pn=[basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5363_2_BI121719/',
    #     basepath + 'GPe/Naive/CAG/Arch/Right/5x30/AG4486_2_BI052819/', #File from low res camera
    #     basepath + 'Str/Naive/A2A/Ai32/Bilateral/10x10/AG5477_4_BI022620/',
    #     ]
    if avg_mean_model == None:
        avg_mean_model = np.poly1d([0.01197452, 0.27856233, 8.09341638])
    if avg_std_model == None:
        avg_std_model = np.poly1d([ 0.01126615,  0.27524898, 10.82751783])
    #Perform a model-based z-scoring:
    y=y-avg_mean_model(x)
    y=y/avg_std_model(x) # Predicted std. as opposed to actual
    return y

def binary_vector_score_event_accuracy(target,pred, est_tn = True):
    hit_rate = sum((pred ==1) & (target==1)) / sum(target==1)
    miss_rate = sum((pred == 0) & (target ==1)) / sum(target==1)
    fa_rate= sum( (pred == 1) & (target == 0)) / sum(target==0)
    print('Raw: %1.3f Hit, %1.3f Miss, %1.3f FA ' % (hit_rate, miss_rate, fa_rate))
    
    #Per target event, detect prediction hits: 
    hit=[]
    on,off=signals.thresh(target,0.5)
    for o,f in zip(on,off):
        ind= pred[o:f]
        if any(ind):
            hit.append(1)
        else:
            hit.append(0)
    
    mean_dur = np.mean(off - on)
    # Detect false alarms
    fa=[]
    on,off=signals.thresh(pred,0.5)
    for o,f in zip(on,off):
        ind = target[o:f]
        if not(any(ind)):
            fa.append(1)
        else:
            fa.append(0)
    
    
    if est_tn == True:
        true_negative = round( np.sum(target==0) / mean_dur)
    else:
        true_negative=len(hit)  # This can be made to be true by adding confirmed true negative events (see below)
    cr = true_negative - sum(fa)
    miss = np.sum(np.array(hit)==0)
    return sum(hit), sum(fa), cr, miss

def tabular_predict_from_nn(tab_fn,weights_fn, xs=None):
     #learner = load_learner(model_fn)
     to_nn=fasttab.load_pickle(tab_fn)
     dls = to_nn.dataloaders(1024)
     learn = fasttab.tabular_learner(dls, metrics=fasttab.accuracy) #BrierScore doesn't seem to work with lr_find()
     learn.load(weights_fn)
     if not isinstance(xs, pd.DataFrame):
         return learn
     
     dl = learn.dls.test_dl(xs, bs=64) # apply transforms
     preds,  _ = learn.get_preds(dl=dl) # get prediction
     return preds
 
def tabular_predict_from_rf(rf_model_fn,xs):
    m=joblib.load(rf_model_fn)
    # pdb.set_trace()
    pred = m.predict(xs)
    return pred

def nn_rf_predict_from_raw(raw,meta,prob_thresh=0.5,low_pass_freq=1, 
                       weights_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/bi_rearing_nn_weightsv3',
                       tab_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/to_nnv3.pkl',
                       rf_model_fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/rf_model_v3.joblib'
                       ):
    '''
        Specify an experiment folder (ffn) and the observation name to use in the
        Rearing observations.boris file with human scored rearing data in that folder.
        Assumes many other files are in that folder (raw csv, metatdata, etc,) 
        and that videos have already been both: 1) scored by a human observer and 2)
        pre-processed with deeplabcut.
    '''
    
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
                  'side_length_px_detrend',] #'dlc_front_over_rear_length', #v3 and later should exclude 'dlc_front_over_rear_length'

    
    dep_var = 'human_scored_rear'
    raw =  raw_to_prediction_format(raw,meta,
                                     rescale = True,
                                     avg_model_detrend = True,
                                     z_score_x_y = True,
                                     flip_y = True)
    dat = raw.loc[:,use_cols] 
    # pdb.set_trace()
    if 'Unnamed: 0' in dat.columns:
        dat.drop('Unnamed: 0',axis = 1, inplace =True)
    dat.fillna(method = 'ffill',inplace = True)
    dat.fillna(method = 'bfill',inplace = True)
    pred = tabular_predict_from_nn(tab_fn,weights_fn, xs=dat)
    
    if isinstance(rf_model_fn,str):
        ensembling = True
        xs = dat.drop(dep_var,axis = 1)
        rf_pred = tabular_predict_from_rf(rf_model_fn, xs)
        raw['rf_pred']=rf_pred
    else:
        ensembling = False
    
    raw['nn_pred'] = pred[:,1]
    
    if ensembling == True:
        final_pred = ( raw['nn_pred'] + raw['rf_pred']) / 2
    else:
        final_pred = raw['nn_pred']
     
        
    #Lowpass filter prediction:
    if not (low_pass_freq == None):
        lp_final_pred= signals.pad_lowpass_unpad(final_pred,
                                             low_pass_freq,
                                             meta['fs'][0])
    else:
        lp_final_pred = final_pred
    return final_pred, lp_final_pred > prob_thresh

def rear_nn_auroc_perf(ffn,boris_obs,prob_thresh=0.5,low_pass_freq=None, 
                       weights_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/bi_rearing_nn_weightsv2',
                       tab_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/to_nnv2.pkl',
                       rf_model_fn = None
                       ):
    '''
        Specify an experiment folder (ffn) and the observation name to use in the
        Rearing observations.boris file with human scored rearing data in that folder.
        Assumes many other files are in that folder (raw csv, metatdata, etc,) 
        and that videos have already been both: 1) scored by a human observer and 2)
        pre-processed with deeplabcut.
    '''
    
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
              'side_length_px_detrend',]

    
    dep_var = 'human_scored_rear'
    raw,meta = ethovision_tools.csv_load(dataloc.raw_csv(ffn),columns=['time'],method='raw')
    boris_fn = Path(ffn).joinpath('Rearing Observations.boris')
    f = open(boris_fn,"r")
    boris= json.loads(f.read())
    f.close()
    dat =  combine_raw_csv_for_modeling([ffn],[boris_obs],
                                     use_cols,rescale = True,
                                     avg_model_detrend = True,
                                     z_score_x_y = True,
                                     flip_y = True)
    if 'Unnamed: 0' in dat.columns:
        dat.drop('Unnamed: 0',axis = 1, inplace =True)
    dat.fillna(method = 'ffill',inplace = True)
    dat.fillna(method = 'bfill',inplace = True)
    pred = tabular_predict_from_nn(tab_fn,weights_fn, xs=dat)
    
    if isinstance(rf_model_fn,str):
        ensembling = True
        xs = dat.drop(dep_var,axis = 1)
        rf_pred = tabular_predict_from_rf(rf_model_fn, xs)
        raw['rf_pred']=rf_pred
    else:
        ensembling = False
    # pdb.set_trace()

    
    # 
    human_rear_score = behavior.boris_to_logical_vector(raw,boris,boris_obs,'a','d')
    raw['human_scored']=human_rear_score
    raw['nn_pred'] = pred[:,1]
    
    if ensembling == True:
        raw['final_pred'] = ( raw['nn_pred'] + raw['rf_pred']) / 2
    else:
        raw['final_pred'] = raw['nn_pred']
        
    #Lowpass filter prediction:
    if not (low_pass_freq == None):
        raw['final_pred']= signals.pad_lowpass_unpad(raw['final_pred'],
                                                    low_pass_freq,
                                                    meta['fs'][0])
    

    b,r,m = ethovision_tools.boris_prep_from_df(raw, meta, plot_cols=['time','final_pred','human_scored'],
                                        event_col=['final_pred'],event_thresh=prob_thresh,
                   )
    auroc = metrics.roc_auc_score(raw['human_scored'].values.astype(np.int16),
                                  raw['final_pred'].values)
    print('%1.5f AUROC' % auroc)
    return auroc, raw['final_pred'], human_rear_score