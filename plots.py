#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:37:06 2020

@author: brian
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import gca 
from matplotlib.collections import PatchCollection
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.patches as mpatches
from gittislab import behavior
from scipy.stats import ttest_rel
def get_subs(axes):
    dd=1
    for d in axes.shape:
        d = dd*d
        dd=d
    ind=[]
    for i in range(0,d,1):
        row,col = np.unravel_index(i,axes.shape,'F') 
        ind.append((row,col))
    return ind

def cleanup(axes):
    inds=get_subs(axes)
    for ind in inds:
        axes[ind].spines["top"].set_visible(False)
        axes[ind].spines["right"].set_visible(False)

def p_to_star(p):
    levels=[0.05, 0.01, 0.001, 0.0001]
    if p <= 0.05:
        star=''
        for crit in levels:
            if p <= crit:
                star += '*'        
    else:
        star = 'n.s.'
    return star

def sig_star(x,p,ax=None,):

    if ax == None:
        ax= gca()
    y=[]  
    for line in ax.lines:
        y.append(max(line.get_ydata()))
    max_y=max(y) 
    max_y *= 1.1
    ax.plot(x,[max_y, max_y],'k')
    mid=sum(x)/2
    max_y *= 1.1
    star=p_to_star(p)
    ax.text(mid,max_y,star,horizontalalignment='center')
    
def mean_cont_plus_conf(clip_ave,xlim=[-45,60],highlight=None,hl_color='b',ax=None):
    '''
    Parameters
    ----------
    clip_ave : DICT with fields:
        'cont_y' Array of continous data (rows=samples,cols=trials) 
        'cont_x' Array of x-values matching rows in 'cont_y'
        'cont_y_conf' Array of y +/- confidence interval for 'cont_y', 
            col 0 = upper bound (y + conf)
            col 1 = lower bound (y - conf)
        See: Output from gittislab.behavior.stim_clip_average() for example 
            generation of this dict.
    xlim : LIST, optional
        x axis bounds to plot, in seconds. The default is [-45,60].
    highlight : LIST, optional
        3-value list describing rectangular patch to highlight on plot.
            [start x, stop x, height y]. The default is None. (No patch)
            E.g. [0,30,15] -- plot a rectangle from 0 to 30s, 15 high
    hl_color  : STR, optional
        string specifying color, default = 'b' (blue)
    ax : matplotlib.pyplot.subplot axis handle if plotting in subplot
    Returns
    -------
    fig : matplotlib.pyplot.subplot figure handle
        DESCRIPTION.
    ax : matplotlib.pyplot.subplot axis handle
        DESCRIPTION.

    '''
    y=clip_ave['cont_y']
    x=clip_ave['cont_x']
    if ax == None: #If no axis provided, create a new plot
        fig,ax=plt.subplots()
    
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    
    #If highlight patch box specified, plot it first:
    if highlight:
        ax.fill_between(highlight[0:2],[highlight[2],highlight[2]],[0,0],
                        hl_color, alpha=0.3,edgecolor='none')
        
    # Plot confidence interval:
    conf_int=clip_ave['cont_y_conf']
    ub = conf_int[:,0]  #Positive confidence interval
    lb = conf_int[:,1]  #Negative confidence interval
    ax.fill_between(x, ub, lb, color='k', alpha=0.3, edgecolor='none',
                    zorder=5)
    
    #Plot mean on top:
    ax.plot(x,y,'k')
    plt.xticks(np.arange(-60,90,10))
    plt.yticks(np.arange(-0,20,5))
    plt.xlim(xlim)
    
    #Set major and minor tick labels
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    plt.ylim([0, 20])
    
    if ax == None:
        return fig,ax
    else:
        return ax

def mean_disc_plus_conf(clip,xlabels,ax=None):
    #Barplot +/- conf
    clip_ave=behavior.stim_clip_average(clip)
    if ax == None:
        fig,ax = plt.subplots()
        axflag=False
    values= [d[0] for d in clip_ave['disc_m']]
    conf= [d[0] for d in clip_ave['disc_conf']]
    #bar plot with errorbars:
    ax.bar(xlabels, values, yerr=conf)
    
    #Perform paired t-test:
    t,p = ttest_rel(clip['disc'][:,0],clip['disc'][:,1])
    # sig_star([0,1],p,ax)
    print(str(p))
    #Axis labels (here or outside of this to be more generalized)
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    if axflag == False:
        return fig, ax
    else:
        return ax
    
# Below 2 functions might be better in ethovision_tools ?
def etho_check_sidecamera(vid_path,frame_array,plot_points=None):
    """
    Save sets of selected frames as a .pngs for closer inspection

    Parameters
    ----------
    vid_path :  pathlib path object
        DESCRIPTION: 
                File path to an ethovision video file (e.g. .mpg videos of paritucular dimension / composition)
                Note: output .pngs will be saved in a subdirectory: vid_path.parent/Frame_output
        
    frame_array :  numpy.ndarray 
        DESCRIPTION:
                n x m array of video frame numbers, such that n figures will be generated, each with m video frames

    plot_points : list of lists (optional)
        DESCRIPTION:
            List of frames containing list of points with [X,Y] (in pixels)
            to plot on that frame. The default is None. 
            see: gen_sidecam_plot_points() for method

    Returns
    -------
    cap : TYPE released cv2 video capture object
        DESCRIPTION.

    """
    
    frame_dir=str(vid_path.parent) + '/Frame_output'
    if os.path.exists(frame_dir)==False:
        os.mkdir(frame_dir)
        print('Made Path')
    cap = cv2.VideoCapture(str(vid_path))
    if (cap.isOpened()== False): 
        print("Error opening video stream %s... skipping attempt" % vid_path)
        return cap
    else:
        width  = cap.get(3) # Pixel width of video
        height = cap.get(4) # Pixel height of video
        fs = cap.get(5) # Sampling rate of video
    for ii,frameset in enumerate(frame_array):
        fig,ax=plt.subplots(1,len(frameset),figsize=(15,5))    
        if any(np.array(frameset)<0):
            print('Negative frame requesteds')
        for i,f in enumerate(frameset):
            cap.set(1,f)
            ret, frame = cap.read()
            if ret == True:
                ax[i].imshow(frame)
                ax[i].set_title('Frame %d, %2.1fs in' % (f,f/fs))
                ax[i].set_xlim(width/2,width)
                if height <= 480:
                    ax[i].set_ylim(0,height/2)
                    ax[i].invert_yaxis()
                if plot_points != None:
                    parts=plot_points[ii][i]
                    for part in parts:
                        ax[i].plot(part[0],part[1],'.r',markersize=3)
            else:
                ax[i].set_title('No frame returned.')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(frame_dir + '/frameset_%03d.png' % ii )
        plt.close()
    cap.release()
    
def gen_sidecam_plot_points(df,parts,framesets):
    """
    Generate list of points to plot on frames for plots.etho_check_sidecamera()

    Parameters
    ----------
    df : pandas dataframe (deeplabcut output)
        DESCRIPTION: 
            imported data from deeplabcut analysis .h5 file output
            
    parts : list of strings
        DESCRIPTION:
             Used to identify columns in df. Corresponds to 'bodyparts' column part.
            
    Returns
    -------
    plot_points: List of frames containing list of points with [X,Y] (in pixels)
            see: plots.etho_check_sidecamera() for useage

    """
   
    valid_parts=np.unique([col[1] for col in df.columns])
    dims=['x','y']
    plot_points=[]
    for frameset in framesets:
        frame_points=[]
        for frame in frameset:
            plot_part=[]
            for part in parts:  
                if part not in valid_parts:
                    print('Invalid part %s requested.' % part)
                else:
                    point=[] #Each ROW of temp will be a point plotted on each frame of frameset
                    for dim in dims:
                        point.append(df[(df.columns[0][0],part,dim)][frame])
                    plot_part.append(point)
            frame_points.append(plot_part)
        plot_points.append(frame_points)
    return plot_points