#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:09:24 2020
Loads .mat file generated in quick_BI021519.m 
@author: brian
"""

import os
import sys
import statistics
from scipy import stats
sys.path.append('/home/brian/Dropbox/Python')
from gittislab import mat_file, signals
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
plot_any=False
use_inline=False

pn='/home/brian/Dropbox/Gittis Lab Data/Electrophysiology/GPe/Naive/A2A/AI32/AG4049_1_BI021519/'
paths=os.listdir(pn)
fns=[]


for p in paths:
    if '_raw.mat' in p:
        fns.append(p)
        print(p)
        
# %% Collect array (dataframe?)
summary=np.zeros((100,5))
last_ii=0
for j,fn in enumerate(fns):
    #fn='BI021519d_raw.mat'
    dat=mat_file.load(os.path.join(pn,fn))
    all_spikes=dat['spiketimes'][0]
    unit_ids=dat['new_assigns'][0]
    spike_ids=dat['assigns'][0]
   
    x = np.array(dat['vel_time'])
    y= np.array(dat['vel'])
        
    #For each unit in spikes.new_assigns
    xms=x*1000
    stims=[]
    for i,stim_time in enumerate(dat['stim_times']):
        st=stim_time
        sd=dat['stim_dur'][0] * 1000
        stims.append(xms[(xms>st) & (xms <= (st+sd))])
        
    thresh=0.01
    is_stim = np.in1d(xms,stims)[...,None]

    fs=4000 #Re derive?
    not_run_ind=np.argwhere((abs(y) < thresh) & ~is_stim)[:,0]
    run_ind=np.argwhere((abs(y) > thresh) &  ~is_stim)[:,0]
    no_run_time=len(not_run_ind) / fs
    run_time = len(run_ind) / fs
    local_i=0
    for (ii,unit) in enumerate(unit_ids,start=last_ii):
        spike_indices=np.argwhere(spike_ids == unit)
        spike_times=all_spikes[spike_indices]
        use_spikes=spike_indices[~np.in1d(spike_times*1000,stims)] #exclude spiking during stimulation
        spike_vel=abs(y[use_spikes])
        summary[ii][0]=j
        summary[ii][1]=unit
        summary[ii][2]=dat['old_assigns'][0][local_i]
        summary[ii][3]=sum(spike_vel > thresh) / run_time
        summary[ii][4]=sum(spike_vel < thresh) / no_run_time
        local_i += 1
    last_ii=ii 
summary=summary[0:last_ii][:]

#%% Plot each unit FR during running vs not:
# if use_inline==True:
#     %matplotlib inline 
# else:
#     print('pop-out Plot')
#     %matplotlib Qt5 
fig, ax = plt.subplots()
# def logmod(x):
#     from math import log10
#     sign=1
#     if x < 0:
#         sign=-1
#     return log10(abs(x)+1)*sign
 
for row in summary:
        # Plot FR during stopped vs. running:
       
        x=[1,2]
        y=[signals.log_modulus(v) for v in row[3:5]]
        ax.plot(x,y,'-ok')
        
        
t,p=stats.ttest_rel(summary[:,3],summary[:,4])
ax.set_ylabel('log (Firing rate (Hz))')
plt.xticks([1,2],['Running','Not running'],rotation=20)

ax.set_title('t-test p= %1.3f' % p)
if use_inline==False:
    mngr = plt.get_current_fig_manager()
    # # to put it into the upper left corner for example:
    mngr.window.setGeometry(0,0,500,500) #Dist from left, dist from top, width, height
    



# %% Write a thresholding function 

def bi_thresh(y,thresh, sign='Pos'):
    import numpy as np
    if sign =='Neg':
        y *=-1
        thresh *=-1
    ind_list=np.concatenate(([0],np.argwhere(y[:,0] > thresh)[:,0]))
    d=np.diff(ind_list) 
    onsets=ind_list[np.argwhere(d > 1) + 1]
    ind_list=np.concatenate(([0],np.argwhere(y < thresh)[:,0]))
    d=np.diff(ind_list)
    offsets=ind_list[np.argwhere(d > 1)+1 ]
    onsets=onsets[:,0]
    offsets=offsets[:,0]
    return onsets, offsets
    
    
def join_crossings(on,off,min_samp):
    import numpy as np
    on=np.concatenate(([0],on))
    off=np.concatenate(([0],off))
    keep_on = []
    keep_off = []
    last_i=0
    for i,o in enumerate(off[1:]):
        if i+1  < len(on) and i >= last_i:
            diff = on[i]-off[i-1]
            if diff > min_samp:
                keep_on.append(i)
                last_i=i
                while last_i < (len(on)-1) and (on[last_i+1] - off[last_i]) < min_samp:
                    last_i +=1
                keep_off.append(last_i)
    return on[keep_on],off[keep_off]
            
# %% Plot onset and offset of running bouts relative to spiking
y= np.array(dat['vel'])
x= np.array(dat['vel_time'])

thresh=0.02
on,off = bi_thresh(abs(y),thresh) 
on,off = join_crossings(on,off,2000) # Bouts must be >500ms apart

#Min bout len:
min_length= 4000/2 # 500ms
keep=off-on > min_length
on=on[keep]
off=off[keep]

# Plot velocity trace
fig, ax = plt.subplots(2,1,sharex='col')
use_x = x < xlim
ax[0].plot(x[use_x],(y[use_x]),'k')  

#Identify periods of running
x_val=x[on]
x_val=x_val[x_val < xlim]
for i,onsets in enumerate(x_val):
    ax[0].plot(onsets,(y[on[i]]),'ro')
    ax[0].plot(x[off[i]],(y[off[i]]),'go')


ax[0].set_ylabel('Speed (arb units)')
ax[0].set_xlabel('Time (s)')
mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
mngr.window.setGeometry(0,2000,4000,500) #Dist from left, dist from top, width, height


# In second subplot plot unit raster
for (ii,unit) in enumerate(unit_ids):
        spike_indices=[i for (i,val) in enumerate(spike_ids) if val == unit]
        spike_times=all_spikes[spike_indices]
        sx=spike_times[spike_times<xlim]
        sy=np.ones(sx.shape) * ii
        ax[1].plot(sx,sy,'|k',markersize=12)

for i,stim_time in enumerate(dat['stim_times']):
    st=stim_time/1000
    sd=dat['stim_dur'][0]
    if st < xlim:
        sx=[st,st+sd]
        sy=[ii+1,ii+1]
        ax[1].plot(sx,sy,'b')

# %% Raster / PSTH example:

thresh=0.03
on,off = bi_thresh(abs(y),thresh) 
on,off = join_crossings(on,off,1000) # Bouts must be >500ms apart

#Min bout len:
min_length= 4000/2 # 500ms
keep=off-on > min_length
on=on[keep]
off=off[keep]
xlim=600
    
#plt.close('all')
def raster(evt_times,spike_times,window,units='ms',ax=None,marker='|k'):
    import numpy as np
    from matplotlib import pyplot as plt
    if ax == None:        
        fig, ax = plt.subplots()
    for i,evt in enumerate(evt_times,start=1):
        ts=spike_times - evt
        ind = (ts > window[0]) & (ts < window[1])
        ts=ts[ind]
        trial_num=np.ones(ts.shape)*i
        ax.plot(ts,trial_num,'%s' % marker,markersize=9)
    ax.set_xlabel('Time (%s)' % units)
    ax.set_ylabel('Trials')    
    ax.yaxis.set_ticks(np.arange(0,len(evt_times),step=5))
    return ax

def psth(evt_times,spike_times,window,nbins=20,units='ms',ax=None):
    import numpy as np
    if ax == None:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
    spike_hist=[]
    for i,evt in enumerate(evt_times,start=1):
        ts=spike_times - evt
        ind = (ts > window[0]) & (ts < window[1])
        ts=ts[ind]
        spike_hist.append(ts)
    ax.hist(np.concatenate(spike_hist),nbins)
    ax.set_xlabel('Time (%s)' % units)
    ax.set_ylabel('Spike Count')    
    return ax

def evt_ensemble(evt_times,x,y,window):
    import numpy as np
    out=[]
    for i,evt in enumerate(evt_times):
        ts=x - evt
        ind = (ts > window[0]) & (ts < window[1])
        ts=ts[ind]
        out.append(y[ind])
    common_len=int(np.median([len(i) for i in out]))
    for i,t in enumerate(out):
        if len(t) > common_len:
            out[i]=t[0:common_len]
        elif len(t) < common_len:
            out.remove(i)
    temp=np.array(out[0][...,None])
    # print(temp.shape)
    for i,col in enumerate(out[1:]):
        temp=np.concatenate((temp,np.array(col[...,None])),axis=1)
    out=temp
    return out


window=[-1000,1000]
use='onsets'
stims=[]
xms=x*1000
for i,stim_time in enumerate(dat['stim_times']):
    st=stim_time
    sd=dat['stim_dur'][0] * 1000
    stims.append(xms[(xms>st) & (xms <= (st+sd))])

for (ii,unit) in enumerate(unit_ids):
    spike_indices=[i for (i,val) in enumerate(spike_ids) if val == unit]
    spike_times=all_spikes[spike_indices] * 1000 #Convert to ms
    if use == 'onsets':
        evt=x[on]*1000
    else:
        evt=x[off]*1000
    fig, ax = plt.subplots(3,1,sharex='col')
    vel_ens=evt_ensemble(evt,x*1000,y,window) *100
    velx=np.arange(window[0],window[1],step=1000/3999.5)
    ax[0].plot(velx,np.mean(abs(vel_ens),axis=1),'r')
    ax[0].set_ylabel('Speed (arb units)')
    raster(evt,spike_times,window,'ms',ax[1])
    
    ax[1].set_markersize=6
    ax[1].set_title('unit %d' % unit)
    raster(evt,stims,window,'ms',ax[1],'_b')
    ax[1].set_markersize=4
    psth(evt,spike_times,window,10,'ms',ax[2])
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(0+ ((ii*200) % 2000),1000,500,800) #Dist from left, dist from top, width, height
 
    
    
    
    
    
    
    
    
    
    
    
    
    