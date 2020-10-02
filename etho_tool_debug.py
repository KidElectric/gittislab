#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:25:12 2020

@author: brian
"""
from gittislab import dataloc
from gittislab import ethovision_tools 
from gittislab import signal, behavior
import os
from pathlib import Path
import numpy as np
import pandas as pd

# inc=['GPi','CAG','Arch','10x30','AG6151_3_CS090720']
inc=[['GPe','CAG','Arch','10x30',]]
exc=[['exclude','_and_Str','Left','Right']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

# %%
pathfn=dataloc.path_to_fn('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPi/Naive/CAG/Arch/',
                          inc=['GPi','zone_1'],exc=['exclude'],filetype='.xlsx')
# %% How to get header:

data=pd.read_excel(pn, sheet_name=0, index_col=0, na_values='-', usecols=[0,1], nrows=36)
data=data.transpose()

# %% How to get hardware items related to stim onsets:
data=pd.read_excel(pn, sheet_name=1, na_values='-', header = 38,usecols=[1,2,3,4,5],)

# %% explore importing parameters from .xlsx raw file:
pn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/10x30/AG3351_2_JS051118/' + \
    'Raw data-BI_two_zone_closed_loop_v2-Trial    45.xlsx')
params=etho_tools.params_from_mat(dataloc.rawmat(pn.parent))


# %%
n=etho_tools.raw_excel_to_h5(pn,force_replace=True)

# %% 
pn=dataloc.rawmat(Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/D1/Arch/Bilateral/10x10_30mW/AG6167_1_BI082420'))
# pn=dataloc.rawmat(Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/10x30/AG3351_2_JS051118/' + \
#     'Raw data-BI_two_zone_closed_loop_v2-Trial    45.xlsx').parent)
# pn=dataloc.rawmat(Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/ChR2/Bilateral/10x30/AG3233_5_BI042518'))
# pn=dataloc.rawmat(Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/PV/Arch/Bilateral/zone_1/AG4187_1_JS011119'))
# pn=dataloc.rawmat(Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/PV/Arch/Bilateral/trig_r/AG4244_1_BI021919'))
mat=mat_file.load(pn)


for i,a in enumerate(mat['params'][0][0]):
    print('%d) %s' % (i,a[0]))

print('')
for i,a in enumerate(mat['noldus'][0][0]):
    if a.size > 0:
        print('%d) %s' % (i,a[0]))
params=ethovision_tools.params_from_mat(pn)
mat_file.dtype_array_to_dict(mat,'noldus')

# %%
for array in mat['noldus']:
    print(array)
    
# %% Raw .mat debug
pn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPi/Naive/CAG/Arch/Bilateral/10x30/AG6151_3_CS090720/Raw_AG6151_3_CS090720.mat')
mat=mat_file.load(pn)
use_keys=[key for key in mat.keys()][3:]
exc=['params','noldus','fs','task','laserOnXY','zone','exp_start','exp_end',
     'task_start','task_stop','laserOffTimes','laserOnTimes',]
for exkey in exc:
    use_keys.remove(exkey)
#Create new dict:
raw_dat={}
for key in use_keys:
    raw_dat[key]=mat[key].flatten()
df=pd.DataFrame.from_dict(raw_dat)

# %%
fields=mat['noldus'].dtype.names
nold={}
for i,f in enumerate(fields):
    temp=mat['noldus'][0][0][i]
    if temp.size >0:
        nold[f]=temp[0]
    else:
        nold[f]=np.nan
        
# %% Smoothing experiment
x_s=signal.boxcar_smooth(raw['x'].values,20)
y_s=signal.boxcar_smooth(raw['y'].values,20)
d=signal.calculateDistance()

# %% etho_tools.unify_h5 debug:
inc=['AG3351_2_JS051118']
exc=['exclude']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_to_h5(basepath,inc,exc)

# %% Test read

raw=pd.read_hdf(pnfn,key='raw')
pp=pd.read_hdf(pnfn,key='params')


# %%
start=time.time()
raw,params,stim=ethovision_tools.raw_params_stim_from_xlsx(pn)
params['fs']=1/np.mean(np.diff(raw['time'][0:]))
for key,value in stim.items():
    params['stim_' + key]=value
dur=time.time()-start
print(dur)

# %% 
start = time.time()
df=pd.read_excel(pn,sheet_name=None)
print(time.time()-start)

# %%
temp=df[[key for key in df.keys()][0]]
header=temp.loc[0:36]
temp=temp.drop([i for i in range(0,37)])
temp=temp.rename({col:temp[col][37] for col in temp.columns},axis='columns')
temp=temp.drop(37)
# %% 
ethovision_tools.unify_to_h5(basepath,inc,exc)