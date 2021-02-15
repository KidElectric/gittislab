#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:25:12 2020

@author: brian
"""

from gittislab import signal, behavior, dataloc, ethovision_tools 
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import pdb
from itertools import compress

# %% RUN BEFORE DLC:
ex0=['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','60min_gpe_muscimol',]]
make_preproc = False
exc=[ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = make_preproc)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% RUN AFTER DLC:
ex0=['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','60min_gpe_muscimol',]]
make_preproc = True
exc=[ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=False,
                                  win=10,make_preproc = make_preproc)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))

# %% 
# inc=['GPi','CAG','Arch','10x30','AG6151_3_CS090720']

inc=['AG','Str','CAG','Arch','10x10_30mW',]
exc=['exclude','_and_Str','Left','Right']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.rawh5(basepath,inc,exc)
df_raw,par_raw=ethovision_tools.h5_load(pns[0])

# %% 10x10 or 10x30 Convert .xlsx to .csv, add in DLC rearing data, & generate preproc *.csv in one go:
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude',
     '_gpe_muscimol','_gpe_pbs','mW','mw']
inc=[['AG','GPe','CAG','Arch','10x'],
     ['AG','Str','A2A','Ai32','10x'],
     ['AG','Str','A2A','ChR2','10x']]
make_preproc = True
exc=[ex0,ex0,ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_raw_to_csv(basepath,
                                  inc,exc,force_replace=True,
                                  win=10,make_preproc =True)

#%% Debug preproc*.csv generation method in zone task:
ex0=['exclude','and_GPe','and_Str','Left','Right',
     'Other XLS','Exclude','zone_1_d1r',
     '_gpe_muscimol','_gpe_pbs','mW','mw']

inc=[['AG','GPe','CAG','Arch','zone_1'],
     ['AG','Str','A2A','Ai32','zone_1'],
     ['AG','Str','A2A','ChR2','zone_1']]

exc=[ex0,ex0,ex0]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.raw_csv_to_preprocessed_csv(basepath,
                                             inc,exc,force_replace=True,
                                             win=10)


# %% unify_h5
inc=[['AG','GPe','FoxP2','ChR2','10x10_20mW',]]
exc=[['exclude','_and_Str','Left','Right']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_to_h5(basepath,inc,exc,force_replace=False,win=10)

# Load a file and plot some vel:
pns=dataloc.rawh5(basepath,inc[0],exc[0])
df_raw,par_raw=ethovision_tools.h5_load(pns[1])
out=behavior.mouse_stim_vel(df_raw,par_raw)
out2=behavior.stim_clip_grab(df_raw,par_raw,raw_col='im')
out3=behavior.stim_clip_grab(df_raw,par_raw,raw_col='im2')
out4=behavior.stim_clip_grab(df_raw,par_raw,raw_col='m2')

trial=3
plt.plot(out['cont'][:,trial],'k')
plt.plot(out2['cont'][:,trial],'r')
plt.plot(out3['cont'][:,trial],'--g')
# plt.plot(out4['cont'][:,trial],'--b')

# %% Unify to CSV is better for a number of reasons, but creates separate metadata and raw .csv files
inc = [['AG',]] #Run on all experiments
exc = [['exclude','Bad','bad','Broken', 'grooming','Exclude','Other XLS']] #'_and_SNr','_and_Str'
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_to_csv(basepath,inc,exc,force_replace=False)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary.stim_dur)
print('Nan stim_durs: %d' % sum(np.isnan(summary.stim_dur)))
print('negative stim durs: %d' % sum((summary.stim_dur<0)))
# summary.iloc[np.where(np.isnan(summary.stim_dur))[0][0],:]
# %%

nan_entries = list(compress(range(len(summary.stim_dur)), 
                            np.isnan(summary.stim_dur)))
for r in nan_entries:
    print('%s %s %s %s' % (summary['anid'][r],summary['proto'][r],
                           summary['exper'][r],summary['settings'][r]))
# %% Print summary of metadata retrieved by query:
# inc = [['AG','A2a','ChR2','Str']] # 'zone_1_30mW'
# exc = [['exclude','_and_SNr','_and_Str','20min_10Hz',
#         'grooming','20min_4Hz','Exclude']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

summary=ethovision_tools.meta_sum_csv(basepath,inc,exc)     
print(summary)
plt.hist(summary.stim_n)
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
# %% Start debugging ZONE task processing
# ethovision_tools.unify_to_h5(basepath,inc,exc)
inc=[['AG','GPe','CAG','Arch','zone_1',]]
exc=[['exclude','_and_Str','Left','Right']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
ethovision_tools.unify_to_csv(basepath,inc,exc)
summary=ethovision_tools.meta_sum_csv(basepath,inc,exc) 

# %% 
xlsx_path=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/'
               + 'Arch/Bilateral/zone_1/AG4766_8_UZ021220/'
               + 'Raw data-bi_two_zone_rm216_v2-Trial     5.xlsx')
raw,params=ethovision_tools.raw_params_from_xlsx(xlsx_path)

# %% Test generalized dataframe analysis function:
a=ethovision_tools.analyze_df(behavior.measure_bearing,basepath,inc,exc)
