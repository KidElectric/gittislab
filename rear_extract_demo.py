#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:35:10 2020

@author: brian
"""

import os
import sys
import statistics
from scipy import stats
sys.path.append('/home/brian/Dropbox/Python')
from gittislab import dataloc
from gittislab import signal
from gittislab import behavior
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.signal import find_peaks 
import time
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

 # %%
 
inc=['GPe','Arch','PV','10x30','Bilateral','AG3474_1'] #1st test
inc=['10x30','A2A','AG3525_10','Bilateral']
exc=[]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior'
paths=dataloc.gen_paths_recurse(basepath,inc,exc,'.h5')
for p in paths:
    print(p)

vid_path=dataloc.video(basepath,inc,exc)
print('Finished')

# %% Faster way:
    
peak,start,stop,df = behavior.detect_rear(paths[0],rear_thresh=0.7,min_thresh=0.2,save_figs=True,
    dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1)

# %% Load and clean DLC data from .h5 file:
    
df = pd.read_hdf(paths[0])

# Clean up rows based on likelihood:
like_thresh=0.1
for col in df.columns:
    if col[2] == 'likelihood':
        ex = df[col] < like_thresh
        xcol=(col[0],col[1],'x')
        ycol=(col[0],col[1],'y')
        df[xcol][ex]=np.nan
        df[ycol][ex]=np.nan

# Cleanup rows based on outliers:
sd_thresh= 4 #sd
for col in df.columns:
    if col[2]=='x' or col[2]=='y':
        m=np.nanmean(df[col])
        sd=np.nanstd(df[col])
        ex=(df[col] > (m + sd_thresh * sd))
        df[col][ex]=np.nan
        ex=(df[col] < (m - sd_thresh * sd))
        df[col][ex]=np.nan

print('Data loaded and cleaned')

# %% Calculate head, front and rear centroids (mean coord):
    
    
exp=df.columns[0][0]
dims=['x','y']

# Head centroid:
use=['snout','side_head']
for dim in dims:
    for i,part in enumerate(use):
        col=(exp,part,dim)
        if i ==0:
            dat=df[col].values[...,None]
        else:
            dat=np.concatenate((dat,df[col].values[...,None]),axis=1)
    new_col=(exp,'head_centroid',dim)
    df[new_col]=np.nanmean(dat,axis=1)

# Front centroid:
use=['snout','side_head','side_left_fore','side_right_fore']
for dim in dims:
    for i,part in enumerate(use):
        col=(exp,part,dim)
        if i ==0:
            dat=df[col].values[...,None]
        else:
            dat=np.concatenate((dat,df[col].values[...,None]),axis=1)
    new_col=(exp,'front_centroid',dim)
    df[new_col]=np.nanmean(dat,axis=1)

# Rear centroid:
use=['side_tail_base','side_left_hind','side_right_hind']
for dim in dims:
    for i,part in enumerate(use):
        col=(exp,part,dim)
        if i ==0:
            dat=df[col].values[...,None]
        else:
            dat=np.concatenate((dat,df[col].values[...,None]),axis=1)
    new_col=(exp,'rear_centroid',dim)
    df[new_col]=np.nanmean(dat,axis=1)


# Mouse body length:
x1=df[(exp,'head_centroid','x')].values
y1=df[(exp,'head_centroid','y')].values
x2=df[(exp,'rear_centroid','x')].values
y2=df[(exp,'rear_centroid','y')].values
col=(exp,'body','length')
mouse_length=[]
for i,r in enumerate(x1):
    mouse_length.append(calculateDistance(x1[i],y1[i],x2[i],y2[i]))
df[col]=mouse_length

# Front over rear:
df[(exp,'front_over_rear','length')]=y2-y1
print('Distances added to dataframe')

# %% Test reading video frame size:
cap=cv2.VideoCapture(str(vid_path[0]))
print(cap.isOpened())
width  = cap.get(3) # float
height = cap.get(4) # float
# %%
fig,ax=plt.subplots(1,3)
#Plot mouse body length(front -> rear centroids) vs body_center y
exp=df.columns[0][0]

#for c in df.columns:
    #print(c[1] +'_' + c[2])
x=df[(exp,'top_body_center','x')].values
y=df[(exp,'top_body_center','y')].values

ax[0].scatter(x[0:10000],y[0:10000],alpha=0.05)
ax[0].set_ylabel('Dist from side cam (px)')
ax[0].set_xlabel('X (px)')
ax[0].set_title('Top-down center-tracking (DLC)')

# Fit polynomial to y vs body_length:

x=(df[(exp,'top_body_center','y')].values)
y=(df[(exp,'front_over_rear','length')].values)

ax[1].scatter(x,y,alpha=0.01)
ax[1].set_ylabel('Head Height (px)')
ax[1].set_xlabel('Dist from side cam (px)')

#Fit polynomial:
# ex = np.isnan(x) | np.isnan(y)
# x=x[~ex]
# y=y[~ex]
# p=np.poly1d(np.polyfit(x,y,2))
# xp=np.linspace(0,300,100)
# ax[1].plot(xp,p(xp),'r')

# Correct mouse length:
# y_cor = y-p(x)
# ax[2].scatter(x,y_cor,alpha=0.01)

# Bin by dist from camera and z-score:
out=[]
step=20
newy=np.zeros(y.shape)
max_val=math.ceil(max(x))
for i,ind in enumerate(range(0,max_val,step)):
    subx=np.argwhere((x>=ind) & (x< (ind + step)))
    suby=y[subx]
    thresh=np.nanmean(suby) + 5 * np.nanstd(suby)
    # out.append(np.mean(suby[suby > thresh]))
    out.append(max(suby))
    # suby=(suby-np.nanmean(suby))/np.nanstd(suby) # Z-score works well but hard to use one thresh across videos
    # if max(suby) > 2:
    #     suby=suby/max(suby)
    # else:
    #     suby=suby/10
    # newy[subx]=suby
xtemp=np.array([i for i in range(0,max_val,step)])+step/2
p=np.poly1d(np.polyfit(xtemp,np.array(out)[:,0],2))
ax[1].plot(xtemp,out,'or')
ax[1].plot(xtemp,p(xtemp),'r')
newy=y/p(x)
ax[2].scatter(x,newy,alpha=0.01)
ax[2].set_ylabel('Corrected Mouse Height')
ax[2].set_xlabel('Dist from side cam(px)')
# ax[3].scatter(x,newy,alpha=0.01)
mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
mngr.window.setGeometry(0,2000,4000,500) #Dist from left, dist from top, width, height

# %% Identify rear onsets and offsets:

fig,ax=plt.subplots(1,2,sharex='row')
fs=29.97
ind=[i for i in range(0,2000)]

# Spline-fitting smooth method:
methods=['spline','gauss']

for ii, method in enumerate(methods):
    ax[ii].plot(np.array(ind)/fs,newy[ind],'b')
    if method == 'spline':
        # coarse=[i for i in range(0,max(ind),1)]
        coarse=ind
        xc=np.array(coarse)/fs
        ytemp=newy[coarse]
        ex=np.isnan(ytemp)
        # s=UnivariateSpline(xc[~ex],ytemp[~ex],s=1)
        s=interp1d(xc[~ex],ytemp[~ex],kind='quadratic')
        smooth_y_loc=s(np.array(ind)/fs)
    elif method == 'gauss':
        smooth_y_loc=gaussian_filter(newy[ind],sigma=3)
    # Peak - detection method:
    min_thresh=0.2
    rear_thresh=0.7
    
    # Find "full rear"
    peaks,start_peak,stop_peak=signal.peak_start_stop(smooth_y_loc,distance=30,
                                                       height=rear_thresh,width=10)
    ax[ii].set_title('Detected %d rears using %s smoothing/interp' % (len(peaks),method))
    ax[ii].plot(np.array(ind)/fs,smooth_y_loc,'r')
    ax[ii].plot([ind[0]/fs, ind[-1]/fs],[rear_thresh,rear_thresh],'--b')
    ax[ii].plot([ind[0]/fs, ind[-1]/fs],[min_thresh,min_thresh],'--b')
    ax[ii].set_ylabel('Corrected mouse height')
    ax[ii].set_xlabel('Time (s)')
    for i,peak in enumerate(peaks[0:10]):
        ax[ii].plot(peak/fs,smooth_y_loc[peak],'go')
        ax[ii].plot(start_peak[i]/fs,smooth_y_loc[start_peak[i]],'bo')
        ax[ii].plot(stop_peak[i]/fs,smooth_y_loc[stop_peak[i]],'bo')
mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
mngr.window.setGeometry(0,2000,4000,500) #Dist from left, dist from top, width, height

# %% Test reading in an image using cv2:
fig,ax=plt.subplots(1,1)
img = cv2.imread('/home/brian/Dropbox/Personal Photos/Baby/1.jpg',0)
plt.imshow(img,cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
ax.set_title('Cutie alert!')
mngr = plt.get_current_fig_manager()
# # to put it into the upper left corner for example:
mngr.window.setGeometry(0,2000,4000,500) #Dist from left, dist from top, width, height

# %%  Full video smooth (SLOW!)
ind=np.array([i for i in range(0,len(newy))])

# # Spline-fitting smooth method (slow!):
# coarse=np.array([i for i in range(0,len(newy),5)])
# ytemp=np.array(newy[coarse])
ex=np.isnan(newy)
# s=UnivariateSpline(coarse[~ex],ytemp[~ex],s=1) #Takes a hecking long time
# smooth_y=s(ind)

s=interp1d(ind[~ex],newy[~ex],kind='quadratic')
smooth_y=s(ind)
# smooth_y=gaussian_filter(newy,sigma=1)
print('finished')
# %% Peak - detection method:


rear_thresh=0.7
min_thresh=0.2
peaks,start_peak,stop_peak = signal.peak_start_stop(smooth_y,height=rear_thresh,min_thresh=min_thresh)
save_img=True
print('Detected %d rears in entire video' % len(peaks))
if save_img==False:
    use_peaks=peaks[0:10]
else:
    use_peaks=peaks
path=str(vid_path[0])
rear_dir=str(vid_path[0].parent) + '/Rears'
if os.path.exists(rear_dir)==False:
    os.mkdir(rear_dir)
    print('Made Path')
cap = cv2.VideoCapture(path)
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
else:
    width  = cap.get(3) # float
    height = cap.get(4) # float
for pp,peak in enumerate(use_peaks):
    
    fig,ax=plt.subplots(1,4,figsize=(11,5))
    # mid=round((offs[ind]+ ons[ind])/2)
    # frames=[ons[ind]-30, ons[ind],mid,offs[ind],offs[ind]+30]
    # frames=[peaks[ind]-30, peaks[ind], peaks[ind]+30]
    frames=[start_peak[pp], peak, stop_peak[pp]]
    if any(np.array(frames)<0):
        print('Negative frame requesteds')
    parts=['front_centroid','rear_centroid']
    dims=['x','y']
    cols=['y.','b.']
    for i,f in enumerate(frames):
        cap.set(1,f)
        ret, frame = cap.read()
        if ret == True:
            ax[i].imshow(frame)
            ax[i].set_title('Frame %d, %2.1fs in' % (f,(f-peak)/fs))
            ax[i].set_xlim(width/2,width)
            if height <= 480:
                ax[i].set_ylim(0,height/2)
            ax[i].invert_yaxis()
            for pn,part in enumerate(parts):
                temp=np.zeros((2,1))
                for ii,dim in enumerate(dims):
                    temp[ii]=df[(exp,part,dim)][f]
                ax[i].plot(temp[0],temp[1],cols[pn],markersize=3)
        else:
            ax[i].set_title('No frame returned.')
    ax[-1].plot(smooth_y[frames[0]:frames[2]],'r')
    mid=frames[1]-frames[0]
    ax[-1].plot(mid,smooth_y[frames[1]],'bo')
    ax[-1].plot(0,smooth_y[frames[0]],'rx')
    ax[-1].plot(frames[2]-frames[0],smooth_y[frames[2]],'gx')
    # cap.release()
    # mngr = plt.get_current_fig_manager()
    # # # to put it into the upper left corner for example:
    # mngr.window.setGeometry(0,2000,4000,500) #Dist from left, dist from top, width, height
    plt.tight_layout()
    plt.show(block=False)
    
    if save_img == True:
        plt.savefig(rear_dir + '/rear_%03d.png' % pp )
        plt.close()
    else:
        plt.savefig('/home/brian/Desktop/test.png') # Save test images
#time.sleep(1)
cap.release()
#time.sleep(5)

# %%
