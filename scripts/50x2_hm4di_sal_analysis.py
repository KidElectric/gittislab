#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:35:29 2021

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

#%% Load in light curve calibration for Dial = 740:
    
#Dial 740 Cal:
fn = Path('/home/brian/Dropbox/Arduino/gittis/blue_laser_power_control_sweep_0-255_arduino_TTL_dial_740.csv')
df_cal=pd.read_csv(fn,sep=';')
thresh=-3

#Dial 740 Cal # 2: 04-30-2021:
# fn=Path('/home/brian/Dropbox/Gittis Lab Hardware/Laser Glow/laser_cal_sweeps/pwm_to_TTL/2021-04-30/blue_laser_power_control_sweep_0-255_arduino_TTL_dial_740_4-30-21.csv')
# df_cal=pd.read_csv(fn,sep=',')
# thresh=-4
y=df_cal.loc[:,'Power (W)'] *1000 # Put in mW


#locate drop offs:
d=np.diff(y)
on,off=signals.thresh(d,thresh,'Neg')


onsets= np.array(off[0:-1])+1
offsets =np.array( off[1:])
plt.figure()
plt.plot(df_cal.loc[:,'Samples '],y)
for o,oo in zip(onsets,offsets):
    plt.plot([o,o],[0,10],'--r')
    plt.plot([oo,oo],[0,10],'--g')
    
newclips=[]
for o,f in zip(onsets,offsets):
    dd= f-o
    if dd > 62:
        o=f-62
    if dd < 62:
        o= o-(62-dd)
        
    t=y[o:f]
    print(len(t))
    newclips.append(t.values)

dat=np.stack(newclips)

x=[x for x in range(0,255,round(255/62))]
x=x[2:]
ax=plots.mean_cont_plus_conf_array(x,dat.T,)
mean_power=np.mean(dat,axis=0)
f = np.polyfit(x,mean_power,deg=2)
laser_cal_fit = np.poly1d(f)
plt.sca(ax)
plt.plot(x,laser_cal_fit(x),'r')

ax.set_ylabel('Blue laser output (mW)')
ax.set_xlabel('Arduino PWM level')
# %% Load in 10x10 day for comparison:
# pnfn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10_hm4di_saline/AG6846_5_BI040121/Raw_AG6846_5_BI040121.csv')
pnfn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/Str/Naive/A2A/Ai32/Bilateral/10x10_hm4di_cno/AG6846_5_BI033021/Raw_AG6846_5_BI033021.csv')
raw,meta=ethovision_tools.csv_load(pnfn,columns='All',method='raw' )
meta['stim_dur']=2
meta['stim_off']=meta['stim_on'] + 2
meta['stim_mean_dur']=2

percentage = lambda x: (np.nansum(x)/len(x))*100
m_clip= behavior.stim_clip_grab(raw,meta,y_col='im', 
                               stim_dur=2,
                               baseline = 2,
                               summarization_fun=percentage)
print(np.mean(m_clip['disc'][:,1]))
# %% 

ex0=['exclude','Bad','GPe','bad','Broken', 'grooming',
 'Exclude','Other XLS']
exc=[ex0]
save = False
plt.close('all')
ans = ['sal','cno',]
# ans = ['cno']
# y_col = 'rear'
y_col = 'vel'
load_method='preproc'
stim_dur = 2
percentage = lambda x: (np.nansum(x)/len(x))*100
rate = lambda x: len(signals.thresh(x,0.5)[0]) / stim_dur

sum_fun = np.mean
for analyze in ans:        
    if analyze == 'sal':
        inc=[['AG','Str','A2A','Ai32','50x2_hm4di_sal',]]
    elif analyze == 'cno':
        inc=[['AG','Str','A2A','Ai32','50x2_hm4di_cno',]]
    else:
        inc=[['AG','Str','A2A','Ai32','50x2_multi_mW',]]
    
    pns=dataloc.raw_csv(basepath,inc[0],ex0)
    if not isinstance(pns,list):
        pns=[pns]
    
    sig_x,sig_y,anid,par = plots.plot_light_curve_sigmoid([pns[0]],
                                                          laser_cal_fit,
                                                          sum_fun,
                                                          y_col=y_col,
                                                          load_method=load_method,
                                                          save=save,iter=100) #Includes bootstrapping 
    if analyze == 'sal':
        sal_x=sig_x
        sal_y=sig_y
        sal_an=anid
        sal_par=np.stack(par)
    elif analyze == 'cno':
        cno_x=sig_x
        cno_y=sig_y
        cno_an=anid
        cno_par=np.stack(par)
    else:
        oth_x=sig_x
        oth_y=sig_y
        oth_an=anid
# %% Compare 2 fits
use_sal=cno_an
# use_sal =[ 'AG6611_7', 'AG6846_5',  'AG6845_9', 'AG6611_6', 'AG6846_3']
plt.close('all')
sal_keep=[]
cno_keep=[]
sal_base_keep=[]
cno_base_keep=[]
def find_mid(x,y):
    dy=np.diff(y)
    mid=np.argwhere(np.max(dy) == dy)[0][0] 
    return x[mid],y[mid]
def find_base(x,y):
    ind=np.argwhere(np.array(x)<=0)
    return np.mean(np.array(y)[ind])
for ii,an in enumerate(cno_an):
    plt.figure()
    
    if an in use_sal:
        cont_ind=np.argwhere( [an == i for i in sal_an])[0][0]
        x = sal_x[cont_ind] 
        y = sal_y[cont_ind]
        xm,ym= find_mid(x,y)
        plt.plot(x,y,'b',label='Saline')
        plt.plot(xm,ym,'bo',label='Midpoint')
        
    else:
        cont_ind= np.argwhere( [an == i for i in oth_an])[0][0]
        x=oth_x[cont_ind]
        y=oth_y[cont_ind]
        xm,ym= find_mid(x,y)
        plt.plot(x,y,'b',label='Control')
        plt.plot(xm,ym,'bo',label='Midpoint')
    sal_base_keep.append(find_base(x,y))
    sal_keep.append(xm) 
    xm,ym = find_mid(cno_x[ii],cno_y[ii])
    cno_keep.append(xm)
    cno_base_keep.append(find_base(cno_x[ii],cno_y[ii]))
    plt.plot(cno_x[ii],cno_y[ii],'r',label='CNO')
    plt.plot(xm,ym,'ro',label='Midpoint')
    
    plt.xticks(ticks=range(0,9,1))
    plt.xlabel('Power (mW)')
    plt.ylabel('% Immobile')
    plt.title('%s' % an)
    plt.ylim([0,100])
    plt.legend()
        
# %% Combrine all onto one plot? Norm baseline


d_mid=cno_par[:,1] - sal_par[:,1] #fitted x0 parameter

per_d=d_mid / sal_par[:,1] * 100 #cno percent threshold shift from control
ld = signals.log_modulus(d_mid) #Transform because definitely not normal
_,pval = scistats.ttest_rel(np.zeros(ld.shape),ld)
print('Mid shift p-value = %1.4f (paired t-test)' % pval)


ld_base = d_mid
_,pval = scistats.ttest_rel(np.zeros(ld_base.shape),ld_base)
print('Base p-value = %1.4f (paired t-test)' % pval)

dat=np.stack((sal_par[:,1],cno_par[:,1]),axis=-1)
f,a,h=plots.mean_bar_plus_conf_array(dat, ['Saline','CNO'])
plt.xlabel('50x2 condition')
plt.ylabel('Immobility threshold x0 (mW)')


# %% Examine curve given mouse speed?
m_clip= behavior.stim_clip_grab(raw,meta,y_col='vel', 
                               stim_dur=2,
                               baseline = 2,
                               summarization_fun=np.mean)

y=(m_clip['disc'][:,1])
x=p(o[:,1])
plt.figure()
plt.plot(x,y,'ok')

#Bin per mW and average:
ym=[]
for i in range(0,8,1):
    ind = (x >= i) & (x < (i+1))
    mbin= np.mean(y[ind])
    ym.append(mbin)
xm=[i+0.5 for i in range(0,8,1)]
plt.plot(xm,ym,'-or')
plt.plot(0,np.mean(m_clip['disc'][:,0]),'*b')

# po,pc=model.fit_double_sigmoid(x,y)
po=model.boostrap_model(x,y,model.fit_double_sigmoid,model_method='lm')
ys=model.double_sigmoid(xs, po[0], po[1], po[2],po[3],po[4],po[5])

# po,pc = model.fit_sigmoid(x,y)
# xs=[i/100 for i in range(25,800,1)]
# ys=model.sigmoid(xs, po[0], po[1], po[2],po[3])
plt.plot(xs,ys,'b')
plt.xlabel('Power (mW)')
plt.ylabel('Speed (cm/s)')