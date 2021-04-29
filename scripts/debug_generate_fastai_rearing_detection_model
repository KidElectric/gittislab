#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:58:49 2021

@author: brian
"""
import os
os.environ['PATH'] += os.pathsep + '/home/brian/Dropbox/Python/fastai'
# os.environ['PATH'] += os.pathsep + '/home/brian/Dropbox/Python/fastai/fastbook'

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import pandas as pd
import numpy as np
from pathlib import Path
from gittislab import dataloc, ethovision_tools, signals, plots, behavior, model
import json 
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from dtreeviz.trees import *
# from IPython.display import Image, display_svg, SVG
from matplotlib import pyplot as plt
import joblib
pd.options.mode.chained_assignment = None
pd.options.display.max_rows = 20
pd.options.display.max_columns = 8
dep_var= 'human_scored_rear'
#Base for saving models:
base = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/')



# %% Generate Train and Test .csv files from human scored video data:
#Base for loading data:
basepath = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
version = 4
train_fn='Train_v%d.csv' % version
test_fn='Test_v%d.csv' % version

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

# How to examine possible columns available to use:
# pn=dataloc.raw_csv(test_video_pn[0])
# raw,meta = ethovision_tools.csv_load(pn,method='preproc')
# raw,meta = ethovision_tools.add_dlc_helper(raw,meta,pn.parent)


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
              'side_length_px_detrend','anid'] #'dlc_front_over_rear_length',

meta_to_raw= ['anid']
valid_video_pn = test_video_pn
test_video_boris_obs = test_video_boris_obs

# %%
train = model.combine_raw_csv_for_modeling(train_video_pn,train_video_boris_obs,
                                  use_cols,rescale = True, 
                                  avg_model_detrend = True,
                                  z_score_x_y  = True,
                                  flip_y = True,
                                  meta_to_raw = meta_to_raw)

test =  model.combine_raw_csv_for_modeling(test_video_pn,test_video_boris_obs,
                                 use_cols,rescale = True,
                                 avg_model_detrend = True,
                                 z_score_x_y = True,
                                 flip_y = True,
                                 meta_to_raw = meta_to_raw)


train.to_csv(base/train_fn)

test.to_csv(base/test_fn)
print('\n\nFinished')

# %% Load in data .csv collated by model.py 
df = pd.read_csv(base/train_fn, #V2 stable, trying V3 3/25/21
                low_memory = False)

df.drop(['Unnamed: 0','anid'],axis = 1, inplace =True)
df.fillna(method = 'ffill',inplace = True)
df.fillna(method = 'bfill',inplace = True)
train_len = len(df)
df_test=pd.read_csv(base/test_fn,
                   low_memory = False)
df_test.drop(['Unnamed: 0','anid'],axis = 1,inplace =True)
df_test.fillna(method = 'ffill',inplace=True)
df_test.fillna(method = 'bfill',inplace = True)

df_both=pd.concat([df,df_test])
df_both.reset_index(inplace=True)

idx=df_both.iloc[:,0]
procs = [Categorify, FillMissing]

cond = (df_both.index< train_len )
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))

print('Train = %d == %d' % (len(train_idx),train_len))
print('Test = %d' % len(valid_idx))
print('Sum = %d == Total Len %d ' % (len(train_idx) + len(valid_idx), len(idx)))

cont,cat = cont_cat_split(df, 1, dep_var=dep_var)
to = TabularPandas(df_both, procs, cat, cont, y_names=dep_var, splits=splits)

# %%
xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y
m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);
draw_tree(m, xs, size=7, leaves_parallel=True, precision=2)
