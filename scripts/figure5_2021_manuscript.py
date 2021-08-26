#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 11:22:27 2021
Figure panels associated with A2a ChR2 stim in GPe (low light etc.)
@author: brian
"""


from gittislab import signals, behavior, dataloc, ethovision_tools, plots, model
import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import pandas as pd
import math
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import pdb
from itertools import compress
import seaborn as sns


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
    savepath = Path('/home/brian/Dropbox/Manuscripts/Isett_Gittis_2021/Figure 5')


# %% Preprocess if necessary:
    
ethovision_tools.unify_raw_to_csv(basepath, inc, exc, force_replace=False,
                                  win=10,make_preproc = False)

ethovision_tools.raw_csv_to_preprocessed_csv(basepath,inc,exc,
                                             force_replace=False,win=10)

# %% Plot 0.25mW GPe stim openloop mouse summary with statistics

analysis = 'GPe_A2a_ChR2_0p25mw'
behavior_str = '10x'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,behavior_str=behavior_str)
data = behavior.open_loop_summary_collect(basepath,inc,exc,update_rear=True)
smooth_amnt= [33, 33*3] #Pass 2 here because there are 4 mice w/ 10x30 and 5 mice with 10x10
# fig,stats=plots.plot_openloop_mouse_summary(data,smooth_amnt=smooth_amnt)

# %% Plot zone day summary for 0.25mw:   
analysis = 'GPe_A2a_ChR2_0p5mw'
behavior_str = 'zone_' #'10x'
inc,exc,color,example_mouse = dataloc.experiment_selector(analysis,behavior_str)
data = behavior.zone_rtpp_summary_collect(basepath,inc,exc)
plots.plot_zone_mouse_summary(data,color='k',example_mouse=example_mouse)

# %% Gather openloop data related to immobility and stim intensity
analyses = ['GPe_A2a_ChR2_0p25mw',
            'GPe_A2a_ChR2_0p5mw',
            'GPe_A2a_ChR2_1mw',
            'GPe_A2a_ChR2_2mw']

behavior_str = '10x'
use_fields=['stim_speed','amb_speed',
            'per_mobile',
            'amb_bout_rate','im_bout_rate','rear_bout_rate']
out = {'anid':[],
       'cell_area_opsin':[],
       'stim_dur':[],
       'light_power':[],
       'norm_stim_speed':[],
       'norm_per_mobile': [],
       'norm_amb_bout_rate':[],
       'norm_rear_bout_rate':[],
       'norm_im_bout_rate':[], 
       'norm_amb_speed': []}

for a in analyses:

    inc,exc,color,example_mouse = dataloc.experiment_selector(a, behavior_str=behavior_str)
    data = behavior.open_loop_summary_collect(basepath, inc, exc, 
                                              update_rear=True,
                                              stim_analyze_dur=10)
    summary = ethovision_tools.meta_sum_csv(basepath, inc, exc) 
        
    for index,row in data.iterrows():
        out['anid'].append(row.anid)
        out['light_power'].append(summary.loc[index,'light_power'])
        out['cell_area_opsin'].append(row.cell_area_opsin)
        out['stim_dur'].append(row.stim_dur)
        for f in use_fields:
            val=row[f].flatten()
            norm_val = (val[1]-val[0]) / (val[1]+val[0]) 
            out['norm_'+f].append(norm_val)
        
df = pd.DataFrame(out)
df.to_csv(savepath.joinpath('norm_a2a_chr2_gpe_motor_by_light.csv'))


# %% Gather GPe zone data related to immobility and stim intensity
analyses = ['GPe_A2a_ChR2_0p25mw',
            'GPe_A2a_ChR2_0p5mw',
            'GPe_A2a_ChR2_1mw',
            'GPe_A2a_ChR2_2mw']

behavior_str = 'zone_1'
use_fields=['per_time_z1','per_time_z2']
out = {'anid':[],
       'cell_area_opsin':[],
       'proto':[],
       'stim_dur':[],
       'light_power':[],
       'norm_per_time_z1': [],
       'norm_per_time_z2': [],
       'norm_per_time_sz': []}

for a in analyses:
    temp = a.split('_')[-1].split('m')[0]
    if 'p' in temp:
        temp=temp.replace('p','.')
    intensity= float(temp)
    inc,exc,color,example_mouse = dataloc.experiment_selector(a,behavior_str=behavior_str)
    data = behavior.zone_rtpp_summary_collect(basepath,inc,exc,
                                              stim_analyze_dur='mean',
                                              zone_analyze_dur=10*60)
    for index,row in data.iterrows():
        out['anid'].append(row.anid)
        out['light_power'].append(intensity)
        out['cell_area_opsin'].append(row.cell_area_opsin)
        out['stim_dur'].append(row.stim_dur)
        out['proto'].append(row.proto)
        for f in use_fields:
            val=row[f]
            norm_val = (val[1]-val[0]) / (val[1]+val[0]) 
            out['norm_'+f].append(norm_val)
        if 'zone_1' in row.proto:
            out['norm_per_time_sz'].append(out['norm_per_time_z1'][-1])
        else:
            out['norm_per_time_sz'].append(out['norm_per_time_z2'][-1])
        
df = pd.DataFrame(out)
df.to_csv(savepath.joinpath('norm_a2a_gpe_zone_1_by_light.csv'))
x=df.light_power
y=df.norm_per_time_sz
plt.figure()
plt.plot(x,y,'.k')
plt.ylabel('Norm. SZ Avoidance')
plt.xlabel('Light power (mW)')

# %% Collect A2a: 
    
analyses = ['Str_A2a_ChR2_0p25mw',
            'Str_A2a_ChR2_0p5mw',
            'Str_A2a_ChR2_1mw',
            'Str_A2a_ChR2_2mw']

behavior_str = '10x'
use_fields=['stim_speed','amb_speed',
            'per_mobile',
            'amb_bout_rate','im_bout_rate','rear_bout_rate']
out = {'anid':[],
       'cell_area_opsin':[],
       'stim_dur':[],
       'light_power':[],
       'norm_stim_speed':[],
       'norm_per_mobile': [],
       'norm_amb_bout_rate':[],
       'norm_rear_bout_rate':[],
       'norm_im_bout_rate':[], 
       'norm_amb_speed': []}

for a in analyses:
    temp = a.split('_')[-1].split('m')[0]
    if 'p' in temp:
        temp=temp.replace('p','.')
    intensity= float(temp)
    inc,exc,color,example_mouse = dataloc.experiment_selector(a,behavior_str=behavior_str)
    data = behavior.open_loop_summary_collect(basepath,inc,exc,
                                              update_rear=True,
                                              stim_analyze_dur=10)
    for index,row in data.iterrows():
        out['anid'].append(row.anid)
        out['light_power'].append(intensity)
        out['cell_area_opsin'].append(row.cell_area_opsin)
        out['stim_dur'].append(row.stim_dur)
        for f in use_fields:
            val=row[f].flatten()
            norm_val = (val[1]-val[0]) / (val[1]+val[0]) 
            out['norm_'+f].append(norm_val)
        
df = pd.DataFrame(out)
# df.to_csv(savepath.joinpath('norm_a2a_chr2_str_motor_by_light.csv'))


#%% Collect A2a Zone behavior:
    
analyses = ['Str_A2a_ChR2_0p25mw',
        'Str_A2a_ChR2_0p5mw',
        'Str_A2a_ChR2_1mw',
        'Str_A2a_ChR2_2mw']

behavior_str = 'zone_1'
use_fields=['per_time_z1','per_time_z2']
out = {'anid':[],
       'cell_area_opsin':[],
       'proto':[],
       'stim_dur':[],
       'light_power':[],
       'norm_per_time_z1': [],
       'norm_per_time_z2': [],
       'norm_per_time_sz': []}

for a in analyses:
    temp = a.split('_')[-1].split('m')[0]
    if 'p' in temp:
        temp=temp.replace('p','.')
    intensity= float(temp)
    inc,exc,color,example_mouse = dataloc.experiment_selector(a,
                                                              behavior_str=behavior_str)
    data = behavior.zone_rtpp_summary_collect(basepath,inc,exc,
                                              stim_analyze_dur='mean',
                                              zone_analyze_dur=10*60)
    for index,row in data.iterrows():
        out['anid'].append(row.anid)
        out['light_power'].append(intensity)
        out['cell_area_opsin'].append(row.cell_area_opsin)
        out['stim_dur'].append(row.stim_dur)
        out['proto'].append(row.proto)
        for f in use_fields:
            val=row[f]
            norm_val = (val[1]-val[0]) / (val[1]+val[0]) 
            out['norm_'+f].append(norm_val)
        if 'zone_1' in row.proto:
            out['norm_per_time_sz'].append(out['norm_per_time_z1'][-1])
        else:
            out['norm_per_time_sz'].append(out['norm_per_time_z2'][-1])
        
df = pd.DataFrame(out)
df.to_csv(savepath.joinpath('norm_a2a_str_zone_1_by_light.csv'))
x=df.light_power
y=df.norm_per_time_z1
plt.figure()
plt.plot(x,y,'.k')
plt.ylabel('Norm. SZ Avoidance')
plt.xlabel('Light power (mW)')

#%% Collect D1 Arch 30mw

analyses = ['Str_D1_Arch_1mw',
            'Str_D1_Arch_3mw',
            'Str_D1_Arch_20mw'
            'Str_D1_Arch_30mw']
behavior_str = '10x'
use_fields=['stim_speed','amb_speed',
            'per_mobile',
            'amb_bout_rate','im_bout_rate','rear_bout_rate']
out = {'anid':[],
       'cell_area_opsin':[],
       'stim_dur':[],
       'light_power':[],
       'norm_stim_speed':[],
       'norm_per_mobile': [],
       'norm_amb_bout_rate':[],
       'norm_rear_bout_rate':[],
       'norm_im_bout_rate':[], 
       'norm_amb_speed': []}

for a in analyses:
    temp = a.split('_')[-1].split('m')[0]
    if 'p' in temp:
        temp=temp.replace('p','.')
    intensity= float(temp)
    inc,exc,color,example_mouse = dataloc.experiment_selector(a,behavior_str=behavior_str)
    data = behavior.open_loop_summary_collect(basepath,inc,exc,update_rear=True)
    for index,row in data.iterrows():
        out['anid'].append(row.anid)
        out['light_power'].append(intensity)
        out['cell_area_opsin'].append(row.cell_area_opsin)
        out['stim_dur'].append(row.stim_dur)
        for f in use_fields:
            val=row[f].flatten()
            norm_val = (val[1]-val[0]) / (val[1]+val[0]) 
            out['norm_'+f].append(norm_val)
        
df = pd.DataFrame(out)
df.to_csv(savepath.joinpath('norm_d1_arch_str_motor_by_light.csv'))

#%% Plot im bout rate vs. intesnity: D1 Arch:

str = pd.read_csv(savepath.joinpath('norm_d1_arch_str_motor_by_light.csv'))
conds = [str]
cols = ['g']
use_fields=['stim_speed']
# plt.close('all')
iter = 10
lab = 'D1 Arch'
xs=[i for i in range(0,31,1)]
for f in use_fields:
    plt.figure()
    for df,c in zip(conds,cols):
        x=df.light_power
        y=df.loc[:,'norm_'+ f]        
        plt.plot(x,y,'.'+c)
        # all_po = model.bootstrap_model(x,y,
        #                     model.fit_sigmoid,
        #                     model_method='lm',
        #                     iter = iter, 
        #                     subsamp_by=1)
        # ys=model.sigmoid(xs, all_po[0], all_po[1], all_po[2],all_po[3])
        # plt.plot(xs,ys,c,label=lab)
    plt.ylabel(f)
    plt.ylim()
    plt.xlabel('Green light power (mW)')
plt.legend()
plt.savefig(savepath.joinpath('d1_str_arch_vs_lightpower.png'))

#%% Plot im bout rate vs. intensity: GPe & Str:

gpe = pd.read_csv(savepath.joinpath('norm_a2a_chr2_gpe_motor_by_light.csv'))
str = pd.read_csv(savepath.joinpath('norm_a2a_chr2_str_motor_by_light.csv'))
conds = [str,gpe]
cols = ['b','r' ]
offset=[-0.025,0.025]
labels=['A2a Str','A2a GPe term',]
use_fields=['stim_speed']
# plt.close('all')
iter = 30
xs=[i/4 for i in range(0,9,1)]
for f in use_fields:
    plt.figure()
    for df,c,o,lab in zip(conds,cols,offset,labels):
        x=df.light_power
        y=df.loc[:,'norm_'+ f]   
        all_po = model.bootstrap_model(x,y,
                                    model.fit_sigmoid,
                                    model_method='lm',
                                    iter = iter, 
                                    subsamp_by=1)
        ys=model.sigmoid(xs, all_po[0], all_po[1], all_po[2],all_po[3])
        plt.plot(xs,ys,c,label=lab)
        plt.plot(x+o,y,'.'+c)
    plt.ylabel(f)
    plt.xlabel('Blue light power (mW)')
    plt.legend()
plt.savefig(savepath.joinpath('a2a_str_vs_gpe_vs_lightpower.png'))

#%% Measure difference between 1mw GPe and 1mw Str on speed:
    
gpe = pd.read_csv(savepath.joinpath('norm_a2a_chr2_gpe_motor_by_light.csv'))
str = pd.read_csv(savepath.joinpath('norm_a2a_chr2_str_motor_by_light.csv'))
conds = [str,gpe]
cols = ['b','r' ]
offset=[-0.25,0.25]
labels=['A2a Str','A2a GPe term',]
use_fields=['stim_speed']
# plt.close('all')
iter = 30
xs=[i/4 for i in range(0,9,1)]
light_power = 1.00
t={'norm_stim_speed':[],'label':[],'anid':[]}
for f in use_fields:
    data=[]
    for df,label in zip(conds,labels):
        use = df.light_power <= light_power
        df = df.loc[use,:]
        df = df.groupby('anid').mean().reset_index()    #Average all data by mouse    
        dat = df.loc[:,'norm_'+ f].values
        data.append(dat)
        t['norm_'+ f] += list(dat)
        t['label'] += list(np.tile(label,len(dat)))
        t['anid'] += list(df.anid.values)
        
        print(len(df))
    fig,a,h=plots.mean_bar_plus_conf_array(data,
                                        xlabels=labels,
                                        color='k',
                                        confidence = 0.95,
                                        paired = False,)
    # plt.xlabel('Blue light power (mW)')
    
    fig.set_size_inches([6,6])
    plt.plot([-0.5,1.5],[0,0],'-k')
    plt.ylim([-0.75, 0.75])
    plt.yticks(ticks=[-0.5,0,0.5])
    # plt.box(on=False,)
    
    
    df=pd.DataFrame(t)
    sns.swarmplot(x='label', y='norm_stim_speed',data=t, orient='v')
    
    plt.title('0.25-%1.2fmW light' % (light_power))
    plt.ylabel('Norm. light-induced speed change')
# plt.savefig(savepath.joinpath('a2a_str_vs_gpe_0.25-%1.2fmw_norm_speed.pdf' % light_power))

#%% Measure difference between 1mw GPe and 1mw Str on avoidance:
gpe = pd.read_csv(savepath.joinpath('norm_a2a_gpe_zone_1_by_light.csv'))
str = pd.read_csv(savepath.joinpath('norm_a2a_str_zone_1_by_light.csv'))
conds = [str,gpe]
labels=['A2a Str','A2a GPe term',]
use_fields=['per_time_sz']
# plt.close('all')
iter = 30
xs=[i/4 for i in range(0,9,1)]
light_power = 1
t={'norm_per_time_sz':[],'label':[],'anid':[]}

for f in use_fields:
    data=[]    
    for df,label in zip(conds,labels):
        use = df.light_power <= light_power #ASSUMES ALL ARE ZONE 1 PROTOCOL for now
        df = df.loc[use,:]
        df = df.groupby('anid').mean().reset_index()        
        dat = df.loc[:,'norm_'+ f].values
        data.append(dat)
        t['norm_'+ f] += list(dat)
        t['label'] += list(np.tile(label,len(dat)))
        t['anid'] += list(df.anid.values)
        print(len(df))
        
    fig,a,h=plots.mean_bar_plus_conf_array(data,
                                        xlabels=labels,
                                        color='k',
                                        confidence = 0.95,
                                        paired = False,)    

    fig.set_size_inches([6,6])
    plt.plot([-0.5,1.5],[0,0],'-k')
    plt.ylim([-0.75, 0.75])
    
    plt.yticks(ticks=[-0.5,0,0.5])
    # plt.box(on=False,)
    
    
    df=pd.DataFrame(t)
    sns.swarmplot(x='label', y='norm_per_time_sz',data=t, orient='v')
    plt.title('0.25-%1.2fmW light' % (light_power))
    plt.ylabel('Norm. light-induced RTPP stim side pref.')
    
plt.savefig(savepath.joinpath('a2a_str_vs_gpe_0.25-%1.2fmw_norm_avoidance.pdf' % light_power))
