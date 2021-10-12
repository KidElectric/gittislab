#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:28:28 2021

@author: brian
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as scistats
from matplotlib import pyplot as plt
from gittislab import signals, plots, behavior, dataloc, ethovision_tools, model
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
        
    basepath = 'F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\'
else:
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'


def mw2pwm(laser_cal_fit,mw):
    xx = np.arange(0,255)
    yy = laser_cal_fit(xx)
    return xx[(yy < (mw+0.01)) & (yy > (mw-0.01))]

import pickle


# %% 
#Dial 740 Cal:
fn = Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/A2A/Ai32/Bilateral/50x2/exclude/2021-08-23-laser_cal/gpe_buttonB6_0-255_11mW_max_cal_cleaned.csv')
df_cal=pd.read_csv(fn)
plt.figure()
y=df_cal.loc[:,' Power(W)'] *1000 # Put in mW
plt.plot([x for x in range(0,len(y))],y)

#locate drop offs:
d=np.diff(y)
on,off=signals.thresh(d,-2,'Neg')
for o in off:
    plt.plot([o,o],[0,33],'--r')

onsets= [2] + off[0:-1]
offsets = on
durs = [f-o for o,f in zip(onsets,offsets)]
use_dur = 3726 #3662
print(durs)
newclips=[]

for o,f in zip(onsets,offsets):
    dd= f-o

    
    if dd > 3500 :
        if dd > use_dur:
            o=f-use_dur

        t=y[o:f]
        
        newclips.append(t.values)

dat=np.stack(newclips)

x=[x for x in np.arange(0,255,255/use_dur)]

# x=x[2:]
# ax=plots.mean_cont_plus_conf_array(x,dat.T,)
mean_power=np.mean(dat,axis=0)
f = np.polyfit(x,mean_power,deg=3)
laser_cal_fit = np.poly1d(f)
# plt.sca(ax)
plt.figure()
plt.plot(x,mean_power,'k')
plt.plot(x,laser_cal_fit(x),'r')
ax=plt.gca()
ax.set_ylabel('Blue laser output (mW)')
ax.set_xlabel('Arduino PWM level')

# %% Plot sigmoids

ex0=['exclude','Bad','Str','bad','Broken', 'grooming',
 'Exclude','Other XLS']
exc=[ex0]
save = True
plt.close('all')
y_col = 'vel'
load_method='preproc'
stim_dur = 2
percentage = lambda x: (np.nansum(x)/len(x))*100
rate = lambda x: len(signals.thresh(x,0.5)[0]) / stim_dur

sum_fun = np.mean
inc=[['AG','GPe','A2A','Ai32','50x2',]]

    
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
button_base = Path(basepath).joinpath('GPe/Naive/A2A/Ai32/Bilateral/50x2/exclude/2021-08-23-laser_cal/')
button_dict={'AG7128_4':button_base.joinpath('gpe_buttonB6_0-255_11mW_max_cal_cleaned_model.pkl'),
             'AG7128_5':button_base.joinpath('gpe_buttonB3_0-255_11mW_max_cal_cleaned_model.pkl'),
             'AG7192_2':button_base.joinpath('gpe_buttonB11_0-255_11mW_max_cal_cleaned_model.pkl'),
             'AG7192_3':button_base.joinpath('gpe_buttonB13_0-255_11mW_max_cal_cleaned_model.pkl'),
             'AG7192_4':button_base.joinpath('gpe_buttonB8_0-255_11mW_max_cal_cleaned_model.pkl')}
#For better precision, load each mouse's calibration file separately.

for pn in pns:
    meta=pd.read_csv(dataloc.meta_csv(basepath,inc=[str(pn.parent)]))
    with open(button_dict[meta.anid[0]],'rb') as pickle_file:
        laser_cal_fit=pickle.load(pickle_file)
 
    sig_x,sig_y,anid,par = plots.plot_light_curve_sigmoid([pn],
                                                          laser_cal_fit,
                                                          sum_fun,
                                                          y_col=y_col,
                                                          load_method=load_method,
                                                          save=save,iter=100) #Includes bootstrapping 

    