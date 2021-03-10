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


def combine_raw_csv_for_modeling(raw_pns,
                                 boris_obs_names,
                                 use_cols,
                                 uniform_boris_fn='Rearing Observations.boris'):
    
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
            vid_res= np.ones((dlc.shape[0],1))  * 704 #Video width 704 x 480
        else:
            vid_res = np.ones((dlc.shape[0],1)) * 1280 #Video width                            
        dlc['video_resolution'] = vid_res
        dlc['human_scored_rear'] = human_scored_rearing
        dlc['head_hind_px_height']= dlc ['dlc_rear_centroid_y'] - dlc['dlc_side_head_y']
        dlc['front_hind_px_height'] = dlc ['dlc_rear_centroid_y'] - dlc['dlc_front_centroid_y']
        dlc['side_length_px']=signal.calculateDistance(dlc['dlc_front_centroid_x'].values,
                                                       dlc['dlc_front_centroid_y'].values,
                                                       dlc ['dlc_rear_centroid_x'].values,
                                                       dlc ['dlc_rear_centroid_y'].values)
        dlc['top_length_px']=signal.calculateDistance(dlc['dlc_top_head_x'].values,
                                                       dlc['dlc_top_head_y'].values,
                                                       dlc ['dlc_top_tail_base_x'].values,
                                                       dlc ['dlc_top_tail_base_y'].values)
        temp=dlc[use_cols]    
        combined = pd.concat([combined,temp])

    #Add time lags (?)
    combined.reset_index(drop=True, inplace=True)
    return combined