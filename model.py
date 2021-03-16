#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 11:16:15 2021

@author: brian
"""
import pandas as pd
import numpy as np
from pathlib import Path
from gittislab import dataloc, ethovision_tools, signal, plots, behavior
import matplotlib.pyplot as plt
import json 
import pdb

def combine_raw_csv_for_modeling(raw_pns,
                                 boris_obs_names,
                                 use_cols,
                                 uniform_boris_fn='Rearing Observations.boris',
                                 rescale = False,
                                 avg_model_detrend = False):
    
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
        dlc['head_hind_px_height']= dlc ['dlc_rear_centroid_y'] - dlc['dlc_side_head_y']
        dlc['head_hind_5hz_pw'] = signal.get_spectral_band_power(dlc['head_hind_px_height'],
                                                                 meta['fs'][0],4.5,6.5)
        
        dlc['front_hind_px_height'] = dlc ['dlc_rear_centroid_y'] - dlc['dlc_front_centroid_y']
        dlc['snout_hind_px_height'] = dlc ['dlc_rear_centroid_y'] - dlc['dlc_snout_y']
        dlc['side_length_px']=signal.calculateDistance(dlc['dlc_front_centroid_x'].values,
                                                       dlc['dlc_front_centroid_y'].values,
                                                       dlc ['dlc_rear_centroid_x'].values,
                                                       dlc ['dlc_rear_centroid_y'].values)
        dlc['top_length_px']=signal.calculateDistance(dlc['dlc_top_head_x'].values,
                                                       dlc['dlc_top_head_y'].values,
                                                       dlc ['dlc_top_tail_base_x'].values,
                                                       dlc ['dlc_top_tail_base_y'].values)
        #Detrend effect of mouse distance from camera using an average pre-fitted
        #z-score approach:
        if avg_model_detrend == True:
            detrend_cols=['head_hind_px_height',
                          'snout_hind_px_height',
                          'front_hind_px_height',
                          'side_length_px',
                          'top_length_px']
            for col in detrend_cols:
                y=dlc[col]
                x=dlc['x']
                dlc[col]=average_z_score(x,y)

        temp=dlc[use_cols]    
        combined = pd.concat([combined,temp])

    #Add time lags (?)
    combined.reset_index(drop=True, inplace=True)
    return combined

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