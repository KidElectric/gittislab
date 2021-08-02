#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 16:55:36 2021

@author: brian
"""
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from gittislab import signals

# %% Use one value only
pca = PCA(n_components=3,whiten=True)
pn=Path('/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/GPe/Naive/Locomotion/unit_coeff')
coeff=pd.read_csv(pn.joinpath('100ms_bin_velocity_bayes_ridge_coefficients.csv'))
unit_data=pd.read_csv(pn.parent.joinpath('gpe_vel_accel_stim_resp_n194.csv'))

coeff=pd.read_csv(pn.joinpath('100ms_bin_accel_bayes_ridge_coefficients.csv'))
# coeff=pd.read_csv(pn.joinpath('100ms_bin_spkhist_bayes_ridge_coefficients.csv'))

X=coeff.iloc[:,5:].values
# X=abs(X)

# coeff=pd.read_csv(pn.joinpath('100ms_bin_accel_bayes_ridge_coefficients.csv'))
# X=X + abs(coeff.iloc[:,5:].values)

x=[]
for col in coeff.columns[5:]:
    x.append(int(col.split('_')[1]))
pca.fit(X)
scores=pca.transform(X)
type = coeff.columns[5].split('_')[0]

if type == 'v':
    ystr = 'Speed'
elif type == 'a':
    ystr = 'Acceleration'
else:
    ystr = 'Spike History'
    
print('Finished')

# %% 

fig,ax = plt.subplots(2,1,sharex=True)

i=1
ind = np.argsort(scores[:,i])
# ind=ind[-40:-1]
ind=ind[0:40]
dat=X[ind,:]
min = scores[ind[0],i]
max = scores[ind[-1],i]

ax[0].imshow(dat,vmin=-0.15,vmax=0.15,
           aspect='auto',
           origin='lower',
           extent=[x[0], x[-1], min,max])
ax[0].title.set_text('PC%d' % (i+1))
ax[0].set_ylabel('PC2 score')

ax[1].plot(x,np.mean(dat,axis=0),'k')
ax[1].set_xlabel('Time from GPe FR (ms)')
ax[1].set_ylabel('Mean weight')
ax[1].plot([x[0],x[-1]],[0,0],'--k')
ax[1].plot([0,0],[-0.05,0.05],'--k')
plt.xlim([-900,900])
# %% Show filters sorted by PC
#Sort X by PC1, PC2, PC3 scores and plot all
fig,ax=plt.subplots(1,3)

    
for i in range(0,3):
    ind=np.argsort(scores[:,i])
    dat=X[ind,:]
    # plt.figure()
    scale_scores=scores[:,i]-np.min(scores[:,i])
    scale_scores=(scale_scores / np.max(scale_scores))*2 -1
    min = -1
    max = 1
    # min=scores[ind[0],i]
    # max=scores[ind[-1],i]
    ax[i].imshow(dat,vmin=-0.15,vmax=0.15,
               aspect='auto',
               origin='lower',
               extent=[x[0], x[-1], min,max])
    ax[i].title.set_text('PC%d' % (i+1))
    if i == 1:
        ax[i].set_xlabel('Time from spike (ms)')
    if i== 0:
        ax[i].set_ylabel('%s Reg. Weight Scaled PC Score' % ystr)
        
plt.tight_layout()
# plt.savefig('vel_abs_model_weights_sorted_pc1-3.png')


#%% Sort by abs X before 0 vs. after 0
xind1=(np.array(x)<0) & (np.array(x)>-300)
xind2=(np.array(x)>0) & (np.array(x)<300 )
pre=np.sum(X[:,xind1],axis=1)
post=np.sum(X[:,xind2],axis=1)
ind=np.argsort(pre-post)
max_ind=np.zeros((X.shape[0],1))
for i,row in enumerate(X[:,(xind2 | xind1)]):
    m=np.max(row)
    max_ind[i,0]=np.argwhere(row==m)[0][0]
# ind=np.argsort(max_ind.flatten())
fig,ax=plt.subplots(1,1)
ax.imshow(X[ind,:],vmin=-0.1,vmax=0.1,
           aspect='auto',
           origin='lower',
           extent=[x[0], x[-1], min,max])
plt.plot([0,0],[-1,1],'--w')
plt.xlim([-250,250])

# %% Compare average weights in top & bottom PC score groups:
thresh=0.5
fig,ax=plt.subplots(1,3)
for i in range(0,3):
    std=np.std(scores[:,i])
    m=np.mean(scores[:,i])
    pos=scores[:,i] > (m + std*thresh)
    neg=scores[:,i] < (m - std*thresh)
    ax[i].plot(x,np.mean(X[neg,:],axis=0),'b')
    ax[i].plot(x,np.mean(X[pos,:],axis=0),'r')
    ax[i].plot([x[0],x[-1]],[0,0],'--k')
    ax[i].plot([0,0],[-0.15, 0.15],'--k')
    ax[i].title.set_text('PC%d' % (i+1))
    ax[i].set_ylabel('Vel. Weight')
    ax[i].set_xlabel('Time from spikes (ms)')
# plt.savefig('vel_abs_model_weights_top_vs_bottom_scores.png')

#%% Compare t0 weight and previously calculated vel tun slope
p = unit_data['vel_tune_slope'].values > 0
n = unit_data['vel_tune_slope'].values < 0
sig = unit_data['vel_tune_pval'].values < 0.05
t0=X[:,np.array(x)==0]
plt.figure()
xx=signals.log_modulus(unit_data['vel_tune_slope'].values)
yy=t0
plt.plot(xx[sig],yy[sig],'.k')

# %% Compare weights that have pos vel tuning vs. neg vel tuning
p = unit_data['vel_tune_slope'].values > 0
n = unit_data['vel_tune_slope'].values < 0
sig = unit_data['vel_tune_pval'].values < 0.05

use = pos & sig
fig,ax=plt.subplots(1,2)
for i in range(0,1):
    std=np.std(scores[:,i])
    m=np.mean(scores[:,i])
    pos= p & sig
    neg= n & sig
    ax[i].plot(x,np.mean(X[neg,:],axis=0),'b')
    ax[i].plot(x,np.mean(X[pos,:],axis=0),'r')
    ax[i].plot([x[0],x[-1]],[0,0],'--k')
    ax[i].plot([0,0],[-0.15, 0.15],'--k')
    # ax[i].title.set_text('PC%d' % (i+1))
    ax[i].set_ylabel('Vel. Weight')
    ax[i].set_xlabel('Time from spikes (ms)')
# plt.savefig('vel_abs_model_weights_top_vs_bottom_scores.png')
# %% Plot PC1 vs PC2, Highlight str_inh vs. str_exc
plt.figure()
plt.plot(scores[:,0],scores[:,1],'.k')
ind0 = unit_data['str_exc'].astype(bool)
plt.plot(scores[ind0,0],scores[ind0,1],'.r')
ind1 = unit_data['str_inh'].astype(bool)
plt.plot(scores[ind1,0],scores[ind1,1],'.b')
a2a=unit_data['A2a'].astype(bool)

plt.figure()
plt.plot(x,np.mean(X[ind1,:],axis=0),'b')
plt.plot(x,np.mean(X[ind0,:],axis=0),'r')
plt.plot(x,np.mean(X[~(ind0 | ind1) & a2a,:],axis=0),'k')
plt.plot([x[0],x[-1]],[0,0],'--k')
# %% Use vel AND accel:
pca = PCA(n_components=3,whiten=True)
pn=Path('/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/GPe/Naive/Locomotion/unit_coeff')
coeff=pd.read_csv(pn.joinpath('100ms_bin_velocity_bayes_ridge_coefficients.csv'))
X=coeff.iloc[:,5:].values
x=[]
for col in coeff.columns[5:]:
    x.append(int(col.split('_')[1]))
    
coeff=pd.read_csv(pn.joinpath('100ms_bin_accel_bayes_ridge_coefficients.csv'))
X=np.concatenate((X,coeff.iloc[:,1:].values),axis=1)

pca.fit(X)
scores=pca.transform(X)
print('Finished')

# %% Show vel and accel filters sorted:
#Sort X by PC1, PC2, PC3 scores and plot all
# 
for i in range(0,3):
    fig,ax=plt.subplots(1,2,sharex=True, sharey=True)
    ind=np.argsort(scores[:,i])
    min=scores[ind[0],i]
    max=scores[ind[-1],i]
    
    dat=X[ind,0:21]
    ax[0].imshow(dat,vmin=-0.2,vmax=0.2,
               aspect='auto',
               extent=[x[0], x[-1], min,max])
    plt.title('PC%d' % (i+1))
    dat=X[ind,21:]
    ax[1].imshow(dat,vmin=-0.2,vmax=0.2,
           aspect='auto',
           extent=[x[0], x[-1], min,max])
    plt.tight_layout()
# %% See if PCs cluster (especially given PREVIOUS analysis! exc or inh by A2a!!!!)
plt.figure()
plt.plot(scores[:,0],scores[:,1],'.k')

# %%

# plt.figure()
# zero_val=X[:,np.array(x)==0][:,0]
# ind=np.argsort(zero_val)
# plt.imshow(X[ind,:],
#                aspect='auto',
#                extent=[x[0], x[-1], zero_val[ind[0]],zero_val[ind[-1]]])