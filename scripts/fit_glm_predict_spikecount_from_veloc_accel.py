#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 12:07:47 2021

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
from gittislab import signals
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input
    https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html
    """
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)
    

# %%
pn=Path('/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/GPe/Naive/Locomotion/unit_csv')
# pp=glob.glob(str(pn.joinpath('*50ms.csv'))) # options: 500ms, 100ms, 50ms (binsize)
# pp=glob.glob(str(pn.joinpath('*frhist_100ms.csv'))) # options: 500ms, 100ms, 50ms (binsize)
pp = glob.glob(str(pn.joinpath('*allfrlags_100ms.csv')))
bin_size = 0.100 # seconds
# For each unit (.csv file), read it into a pandas dataframe

fit_type = 'sc' #In this scenario, use unit firing to predict SPEED
# fit_type = 'vas'
fit_method='lasso'
# fit_method='gauss_basis'
# fit_method='bayes_ridge'
# fit_method= 'ridge'
plotFig= True
saveFig = False
pp=[unit for unit in pp if ('AG6379-9' in unit) \
    and (('unit_006' in unit) or ('unit_009' in unit) \
          or ('unit_015' in unit) or ('unit_010' in unit) )]
fr_lags = False
gauss_filt=signals.gaussian_filter1d(int(2/bin_size), sigma=1)
keep_vel=np.zeros((len(pp),21))
keep_accel=np.zeros((len(pp),21))
keep_spkhist=np.zeros((len(pp),8))
keep_unit_data=[]
for ii,unit in enumerate(pp):
    df=pd.read_csv(unit)    
    unit_info=Path(unit).parts[-1].split('_')
    temp=[int(unit_info[1]),int(unit_info[3]),unit_info[5],unit_info[6]] # unit_num,rec_id,mouse_id,exp_id
    keep_unit_data.append(temp)
    #Set up X:
    vel_cols=[col for col in df.columns if 'v' in col]
    accel_cols=[col for col in df.columns if 'a' in col]
    sc_col=[col for col in df.columns if 'sc' in col]      
    
    #Convert spike counts to smooth inst. FR

    if 'sc' not in fit_type:
        y=df.loc[:,'spike_counts'].values #might only need y column name
        y=sp.signal.convolve(y,gauss_filt, mode='same')/bin_size
        y = (y-np.min(y))
        y = (y/np.max(y))
        
    #Select which data to use for the fit:
    if fit_type == 'vel':
        X = df.loc[:,vel_cols]
        X = X.transform(signals.log_modulus).values
        
    elif fit_type == 'accel':
        X = df.loc[:,accel_cols]
        X = X.transform(signals.log_modulus).values
    elif fit_type == 'vas':
        #Convert spike counts to smooth inst. FR in X:
        
        #Log transform speed and scale from 0 to 1:
        X = df.loc[:,vel_cols].transform(signals.log_modulus)
        X = X - np.min(X)
        X = X / np.max(X)
        
        #Log transform acceleration,
        # and scale from -0.5 to 0.5
        accel = df.loc[:,accel_cols].transform(signals.log_modulus)
        accel =  accel - np.min(accel)
        accel = (accel / np.max(accel)) - 0.5       
        X = np.concatenate((X,accel),axis=1)
        
        # Add in firing rate history
        sc = df.loc[:,sc_col[0:-2]].values / bin_size
        xi = sc.shape[1]
        sc = sc - np.min(sc)
        sc = sc / np.max(sc)
        X = np.concatenate((X,sc),axis=1)
    elif fit_type == 'sc':
        for j,col in enumerate(sc_col):
            dat=df.loc[:,col].values
            t=sp.signal.convolve(dat,gauss_filt, mode='same')/bin_size
            if j ==0:
                X=t[:,np.newaxis]
            else:
                X=np.concatenate((X,t[:,np.newaxis]),axis=1)
        X = signals.log_modulus(X)
        X = X - np.min(X)
        X = X / np.max(X)
        
        #Add in velocity history:
        vel= df.loc[:,vel_cols[0:1]].transform(signals.log_modulus)
        vel -= np.min(vel)
        vel /= np.max(vel)
        X=np.concatenate((X,vel),axis=1)
        y=df.loc[:,'v_0']
        y=signals.log_modulus(y)
    
            
    else: #Predict spike count using all columns
        X = df.iloc[:,1:]
    
    if ii==0:
        x=[]
        for col in vel_cols:
            x.append(int(col.split('_')[1]))
        x=np.array(x)
        
    # xscaler = StandardScaler()
    # X=xscaler.fit_transform(X)
    

    # y=np.diff(np.concatenate((y,y[-1,np.newaxis]),axis=0))
    
    
    # y=signals.log_modulus(y)    
    # y=(y - np.nanmean(y)) / np.nanstd(y)

    
    # Lasso method:
    if fit_method== 'lasso':
        clf = linear_model.Lasso(alpha=1e-4,normalize=False,max_iter = 10000)
        clf.fit(X,y)
        coef = clf.coef_
    elif fit_method == 'gauss_basis':
        clf = make_pipeline(GaussianFeatures(20),
                            linear_model.LinearRegression())
        clf.fit(X[:,:, np.newaxis], y)
        pred = clf.predict(X[:,:, np.newaxis])
    elif fit_method == 'bayes_ridge':
        clf=linear_model.BayesianRidge(compute_score=True)
        clf.fit(X,y)
        coef=clf.coef_
    elif fit_method == 'ridge':
        clf = linear_model.Ridge()
        clf.fit(X,y)
        coef=clf.coef_
    keep_vel[ii,:]=coef[0:len(x)]
    if fit_type == 'vas':
        keep_accel[ii,:]=coef[len(x):(2*len(x))]
        keep_spkhist[ii,:] = coef[2*len(x):]
    title_str=Path(unit).parts[-1].split('.')[0] + '_' + fit_method;
    if plotFig == True:
        plt.figure()
        plt.plot(x,coef[0:len(x)])        
        if fit_type == 'both':
            plt.plot(x,coef[len(x):])
        elif fit_type == 'vas':
            plt.plot(x,keep_accel[ii,:])
            plt.plot(x[0:xi],keep_spkhist[ii,:])
        plt.plot([x[0],x[-1]],[0,0],'--k')
        plt.plot([0,0],[-2,2],'--k')        
        plt.title(title_str)
        plt.ylim([-0.10,0.10])
        plt.ylabel('Influence on t0 spiking (weight)')
        plt.xlabel('Time from spikes (ms)')
        plt.show()
        if saveFig == True:
            plt.savefig(pn.joinpath(title_str))
            plt.close(plt.gcf())
    
    print('%s ... finished' % title_str)
print('Completed')

# %% Save locomotion filters
unit_info_cols=['unit_num', 'rec_id', 'mouse_id','exp_id']
df= pd.DataFrame(data=keep_unit_data, columns = unit_info_cols)
df[vel_cols]=keep_vel
df.sort_values(by='unit_num',axis=0,inplace=True,ignore_index=True)
df.to_csv(pn.joinpath('100ms_bin_velocity_bayes_ridge_coefficients.csv'))

df= pd.DataFrame(data=keep_unit_data, columns = unit_info_cols)
df[accel_cols]=keep_accel
df.sort_values(by='unit_num',axis=0,inplace=True,ignore_index=True)
df.to_csv(pn.joinpath('100ms_bin_accel_bayes_ridge_coefficients.csv'))

df= pd.DataFrame(data=keep_unit_data, columns = unit_info_cols)
df[sc_col[0:xi]]=keep_spkhist
df.sort_values(by='unit_num',axis=0,inplace=True,ignore_index=True)
df.to_csv(pn.joinpath('100ms_bin_spkhist_bayes_ridge_coefficients.csv'))

# %%
#get time bins from column names

plt.figure(),
plt.plot(y)
plt.plot(X[:,0])
pred=clf.predict(X)
plt.plot(pred)
    
# %%
