#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:15:37 2021

@author: brian
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import scipy as sp
from matplotlib import pyplot as plt
from gittislab import signals, utils
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# %%
base = Path('/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/GPe/Naive/Locomotion/')
units=pd.read_csv(base.joinpath('gpe_vel_accel_stim_resp_n194.csv'))

# %%
pn=Path('/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/GPe/Naive/Locomotion/unit_csv')
# pp=glob.glob(str(pn.joinpath('*50ms.csv'))) # options: 500ms, 100ms, 50ms (binsize)
# pp=glob.glob(str(pn.joinpath('*frhist_100ms.csv'))) # options: 500ms, 100ms, 50ms (binsize)
pp = glob.glob(str(pn.joinpath('*allfrlags_500ms.csv')))
bin_size = 0.500 # seconds
# For each unit (.csv file), read it into a pandas dataframe

fit_type = 'sc' #In this scenario, use unit firing to predict SPEED
# fit_type = 'vas'
fit_method='lasso'
# fit_method='gauss_basis'
# fit_method='bayes_ridge'
# fit_method= 'ridge'
plotFig= True
saveFig = False
# pp=[unit for unit in pp if ('AG6379-9' in unit) \
#     and (('unit_006' in unit) or ('unit_009' in unit) \
#           or ('unit_015' in unit) or ('unit_010' in unit) )]
fr_lags = False
gauss_filt=signals.gaussian_filter1d(int(1/bin_size), sigma=1)
keep_vel=np.zeros((len(pp),1))
keep_accel=np.zeros((len(pp),1))
keep_inter=np.zeros((len(pp),1))
keep_spkhist=np.zeros((len(pp),1))
keep_unit_data={'unit_num':[],
                'rec_id':[],
                'mouse_id':[],
                'exp_id':[],
                'speed_coef':[],
                'decel_coef':[],
                'interaction_coef':[]}

fit_seq= False
use_interaction = False
for ii,unit in enumerate(pp):
    df=pd.read_csv(unit)    
    unit_info=Path(unit).parts[-1].split('_')
    temp=[int(unit_info[1]),int(unit_info[3]),unit_info[5],unit_info[6]] # unit_num,rec_id,mouse_id,exp_id
    keep_unit_data['unit_num'].append(int(unit_info[1]))
    keep_unit_data['rec_id'].append(int(unit_info[3]))
    keep_unit_data['mouse_id'].append(unit_info[5])
    keep_unit_data['exp_id'].append(unit_info[6])
    
    # keep_unit_data.append(temp)
    #Set up X:
    vel_cols=[col for col in df.columns if 'v' in col]
    accel_cols=[col for col in df.columns if 'a' in col]
    sc_col=[col for col in df.columns if 'sc' in col]   
    accel = df.loc[:,'a_0'].transform(signals.log_modulus).values
    accel[accel > 0] = 0
    accel = accel - np.min(accel)
    decel = (accel / np.max(accel))  -1 
    
    vel=df.loc[:,'v_0'].transform(signals.log_modulus).values
    vel=np.abs(vel)
    vel=vel-np.min(vel)
    vel=vel/np.max(vel)
    
    inter= vel * accel
    inter = inter - np.min(inter)
    inter = inter / np.max(inter)
    if use_interaction == True:
        X=np.concatenate((vel[:,np.newaxis],
                          decel[:,np.newaxis],
                          inter[:,np.newaxis]),axis=1)
    else:
        X=np.concatenate((vel[:,np.newaxis],
                          decel[:,np.newaxis]),axis=1)     
                         
    y=df.loc[:,'spike_counts'].values #might only need y column name
    # y=sp.signal.convolve(y,gauss_filt, mode='same')/bin_size
    y = (y-np.min(y))
    y = (y/np.max(y))
    
    
    if fit_method== 'lasso':
        
        if fit_seq == True:
            mod_a= linear_model.Lasso(alpha=1e-4,normalize=True,max_iter = 10000)
            mod_a.fit(X[:,0:1],y)
            yy=mod_a.predict(X[:,0:1])
            yy=y-yy
            yy[yy<0]=0
            mod_b = linear_model.Lasso(alpha=1e-4,normalize=True,max_iter = 10000)
            
            mod_b.fit(X[:,1:2],yy)
            coef=np.concatenate((mod_a.coef_,mod_b.coef_),axis=0)
            keep_vel[ii],keep_accel[ii]=coef
            keep_inter[ii]=True
        else:
            clf = linear_model.Lasso(alpha=1e-4,normalize=True,max_iter = 10000)
            clf.fit(X,y)
            coef = clf.coef_
            if use_interaction == True:
                keep_vel[ii],keep_accel[ii],keep_inter[ii]=coef
            else:
                keep_vel[ii],keep_accel[ii]=coef
                keep_inter[ii]=True
    keep_unit_data['speed_coef'].append(keep_vel[ii])
    keep_unit_data['decel_coef'].append(keep_accel[ii])
    keep_unit_data['interaction_coef'].append(keep_inter[ii])
    
    title_str=Path(unit).parts[-1].split('.')[0] + '_' + fit_method
    print(title_str, coef)
fit_table=pd.DataFrame(keep_unit_data)

# %%
test=units.merge(fit_table,on='unit_num')
exp_test=np.sum(test.loc[:,'exp_id_x'] == test.loc[:,'exp_id_y'])
if exp_test < test.shape[0]:
    raise ValueError('Mismatch in experiments after merger.')
mouse_test=np.sum(test.loc[:,'mouse_id_x'] == test.loc[:,'mouse_id_y'])
if mouse_test< test.shape[0]:
    raise ValueError('Mismatch in mouse_ids after merger.')
    
# %% Plot mean velocity vs. FR based on model coeffs

fig = plt.figure()
cols=['b','k','r']
coef_labs=['speed_coef','decel_coef']
xlabs=['Normalized speed (SD)','Normalized decel (SD)']
vel_cols=[p for p in test.columns if 'v_' in p]
decel_cols=[p for p in test.columns if 'a_' in p]
use_cols=[vel_cols,decel_cols]
xlims=[[0,4],[-3,0]]
for j, coef in enumerate(coef_labs):
    ax = fig.add_subplot(1,len(coef_labs),j+1)
    inds=[test.loc[:,coef].values >  0,
          test.loc[:,coef].values == 0,
          test.loc[:,coef].values <  0]
    
    x=[float(p.split('_')[1]) for p in use_cols[j]]
    
    for i,ind in enumerate(inds):
        dat=test.loc[ind,use_cols[j]].values
        m,ci=signals.conf_int_on_matrix(dat,axis=0)
        
        a=plt.fill_between(x,ci[:,0],ci[:,1],color=cols[i])
        a.set_alpha(0.3)
        plt.plot(x,m,cols[i])
    plt.plot([x[0],x[-1]],[0,0],'--k')
    plt.xlabel(xlabs[j])
    plt.ylabel('Firing rate (SD)')
    plt.xlim(xlims[j])
    plt.ylim([-2,2])

# %%
print('Decel only', np.sum((keep_vel ==0) & (keep_accel != 0) & (keep_inter ==0)))
print('Vel only ',np.sum((keep_accel ==0) & (keep_vel != 0) & (keep_inter ==0)))
utils.print_percent(keep_vel !=0,np.zeros(keep_vel.shape)==0,'Speed','all units');
utils.print_percent(keep_vel>0, keep_vel !=0,'Speed pos','all speed');
print('Percent vel pos /all vel >0:', np.sum(keep_vel>0) / np.sum(keep_vel != 0) * 100)
print('Sig vel:',np.sum((keep_vel != 0)))
print('Any sig:',np.sum((keep_vel != 0) | (keep_inter !=0) | (keep_accel !=0)))
print('Sig decel:',np.sum(keep_accel != 0))

utils.print_percent(inh,a2a,'Str Inh','A2a');
utils.print_percent(keep_accel<0, keep_accel !=0,'Decel pos','all decel');
print('Sig interactions ', np.sum(keep_inter !=0))
print('Speed or Decel',np.sum((keep_vel !=0) | (keep_accel != 0)))



# %% Plot 4 conditions as different rings/dots
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(keep_vel,keep_accel,'.k')
plt.plot([0,0],[np.min(keep_accel),np.max(keep_accel)],'--k')
plt.plot([np.min(keep_vel),np.max(keep_vel)],[0,0],'--k')

ap = keep_accel > 0
an = keep_accel < 0
vp = keep_vel > 0
vn = keep_vel < 0

sig_vel= keep_vel != 0
sig_accel= keep_accel !=0
sig_inter = keep_inter !=0


ind = (ap & vp) & (sig_vel | sig_accel )
h=plt.plot(keep_vel[ind],keep_accel[ind],'bo')

ind = (an & vn) & (sig_vel | sig_accel) 
h=plt.plot(keep_vel[ind],keep_accel[ind],'ro')


ind = (ap & vn) & (sig_vel | sig_accel) 

h=plt.plot(keep_vel[ind],keep_accel[ind],'o',
           markerfacecolor='b', markeredgecolor='r',
           markeredgewidth=2)

ind = (an & vp) & (sig_vel | sig_accel) 
h=plt.plot(keep_vel[ind],keep_accel[ind],'o',
           markerfacecolor='r', markeredgecolor='b',
           markeredgewidth=2)

#Vel positive only
ind = (vp) & (sig_vel & ~sig_accel)
# h=plt.plot(keep_vel[ind],keep_accel[ind],'.b')
h=plt.plot(keep_vel[ind],keep_accel[ind],'o',
           markerfacecolor='w',markeredgecolor='b',
           markeredgewidth=2)


#Vel negative only
ind = (vn) & (sig_vel & ~sig_accel)
# h=plt.plot(keep_vel[ind],keep_accel[ind],'.r')
h=plt.plot(keep_vel[ind],keep_accel[ind],'o',
           markerfacecolor='w',markeredgecolor='r',
           markeredgewidth=2)

#Decel negative only
ind = (an) & (~sig_vel & sig_accel)
h=plt.plot(keep_vel[ind],keep_accel[ind],'o',
           markerfacecolor='r',markeredgecolor='w',
           markeredgewidth=2)

#Decel pos only
ind = (ap) & (~sig_vel & sig_accel)
h=plt.plot(keep_vel[ind],keep_accel[ind],'o',
           markerfacecolor='b',markeredgecolor='w',
           markeredgewidth=2)

plt.xlabel('Speed coef')
plt.ylabel('Decel coef')
ax.set_aspect('equal')

# %% Compare A2a ChR2 inhbited vs disinhibited

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(keep_vel,keep_accel,'.k')
plt.plot([0,0],[np.min(keep_accel),np.max(keep_accel)],'--k')
plt.plot([np.min(keep_vel),np.max(keep_vel)],[0,0],'--k')

inh = test.str_inh.values == 1
exc = test.str_exc.values == 1
a2a = test.A2a == True
vp = test.speed_coef > 0
vn = test.speed_coef < 0
ap = test.decel_coef > 0
an = test.decel_coef < 0


plt.plot(test.loc[inh,'speed_coef'], test.loc[inh,'decel_coef'],'mo')
plt.plot(test.loc[exc,'speed_coef'], test.loc[exc,'decel_coef'],'co')
plt.xlabel('Speed coef')
plt.ylabel('Decel coef')
ax.set_aspect('equal')

utils.print_percent(vp & ap & inh,  inh,'Speed pos decel pos','str inh');
utils.print_percent(vp & an & inh, inh,'Speed pos decel neg','str inh');
utils.print_percent(vn & ap & inh, inh,'Speed neg decel pos','str inh');
utils.print_percent(vn & an & inh,  inh,'Speed neg decel neg','str inh');
utils.print_percent(vn & inh,  inh,'Speed neg','str inh');
utils.print_percent(vp & inh,  inh,'Speed pos','str inh');
utils.print_percent(an & inh,  inh,'Decel neg','str inh');
utils.print_percent(ap & inh,  inh,'Decel pos','str inh');

print('')

utils.print_percent(vp & ap & exc,  exc,'Speed pos decel pos','str exc');
utils.print_percent(vp & an & exc, exc,'Speed pos decel neg','str exc');
utils.print_percent(vn & ap & exc, exc,'Speed neg decel pos','str exc');
utils.print_percent(vn & an & exc,  exc,'Speed neg decel neg','str exc');
utils.print_percent(vn & exc,  exc,'Speed neg','str exc');
utils.print_percent(vp & exc,  exc,'Speed pos','str exc');
utils.print_percent(an & exc,  exc,'Decel neg','str exc');
utils.print_percent(ap & exc,  exc,'Decel pos','str exc');

# %% Compare Cag Arch inhbited vs disinhibited

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(keep_vel,keep_accel,'.k')
plt.plot([0,0],[np.min(keep_accel),np.max(keep_accel)],'--k')
plt.plot([np.min(keep_vel),np.max(keep_vel)],[0,0],'--k')

cag = test.CAG_arch.values == 1
inh =cag & (test.stim_fr_change.values < 0) & (test.stim_mod_sig.values==1)
exc =cag & (test.stim_fr_change.values > 0) & (test.stim_mod_sig.values==1)

vp = test.speed_coef > 0
vn = test.speed_coef < 0
ap = test.decel_coef > 0
an = test.decel_coef < 0


plt.plot(test.loc[inh,'speed_coef'], test.loc[inh,'decel_coef'],'mo')
plt.plot(test.loc[exc,'speed_coef'], test.loc[exc,'decel_coef'],'co')
plt.xlabel('Speed coef')
plt.ylabel('Decel coef')
ax.set_aspect('equal')

utils.print_percent(vp & ap & inh,  inh,'Speed pos decel pos','cag inh');
utils.print_percent(vp & an & inh, inh,'Speed pos decel neg','cag inh');
utils.print_percent(vn & ap & inh, inh,'Speed neg decel pos','cag inh');
utils.print_percent(vn & an & inh,  inh,'Speed neg decel neg','cag inh');
utils.print_percent(vn & inh,  inh,'Speed neg','cag inh');
utils.print_percent(vp & inh,  inh,'Speed pos','cag inh');
utils.print_percent(an & inh,  inh,'Decel neg','cag inh');
utils.print_percent(ap & inh,  inh,'Decel pos','cag inh');

print('')

utils.print_percent(vp & ap & exc,  exc,'Speed pos decel pos','cag exc');
utils.print_percent(vp & an & exc, exc,'Speed pos decel neg','cag exc');
utils.print_percent(vn & ap & exc, exc,'Speed neg decel pos','cag exc');
utils.print_percent(vn & an & exc,  exc,'Speed neg decel neg','cag exc');
utils.print_percent(vn & exc,  exc,'Speed neg','cag exc');
utils.print_percent(vp & exc,  exc,'Speed pos','cag exc');
utils.print_percent(an & exc,  exc,'Decel neg','cag exc');
utils.print_percent(ap & exc,  exc,'Decel pos','cag exc');

# %%
np.sum((keep_vel !=0) | (keep_accel != 0))/len(keep_vel)

# %% Save:
    unit_info_cols=['unit_num', 'rec_id', 'mouse_id','exp_id']
df= pd.DataFrame(data=keep_unit_data, columns = unit_info_cols)
df[vel_cols]=keep_vel
df.sort_values(by='unit_num',axis=0,inplace=True,ignore_index=True)
df.to_csv(pn.joinpath('100ms_bin_velocity_bayes_ridge_coefficients.csv'))
