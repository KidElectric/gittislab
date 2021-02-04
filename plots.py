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
import matplotlib.gridspec as gridspec
from gittislab import behavior, ethovision_tools, signal
from scipy.stats import ttest_rel
from scipy.stats import t
import pdb 

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

def sig_star(x,p,max_y,ax=None,):

    if ax == None:
        ax= gca()
    plt.sca(ax)
    max_y *= 1.1
    ax.plot(x,[max_y, max_y],'k')
    mid=sum(x)/2
    max_y *= 1.1
    star=p_to_star(p)
    ax.text(mid,max_y,star,horizontalalignment='center')
    
def connected_lines(x,y,ax=None,color=''):
    '''
    '''
    axflag = True
    if ax == None:
        fig,ax = plt.subplots()
        axflag=False
    plt.sca(ax)
    
    h=[]
    for row in y:
        h0, =ax.plot(x,row,'-o'+color)
        h.append(h0)
    
    if axflag == False:
        return fig, ax, h
    else:
        return ax,h
    
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
    plt.sca(ax)

    
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

def mean_bar_plus_conf(clip,xlabels,use_key='disc',ax=None,clip_method=True,
                       color=''):
    '''
    Use discrete time period summary of clips generated by behavior.stim_clip_grab()
    to plot bar +/- conf. Creat figure /axis and return handles if no axis passed.

    Parameters
    ----------
    clip : TYPE
        DESCRIPTION.
    xlabels : TYPE
        DESCRIPTION.
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    #Barplot +/- conf
    
    axflag = True
    if ax == None:
        fig,ax = plt.subplots()
        axflag=False
    plt.sca(ax)
    if clip_method==True:
        clip_ave=behavior.stim_clip_average(clip,discrete_key=use_key)
        values= [d[0] for d in clip_ave['disc_m']]        
    else:
        values=np.nanmean(clip[use_key],axis=0)
        n=clip[use_key].shape[0]
        confidence = 0.95
        clip_ave={'disc_conf':np.empty((3,1)),}
        for i,data in enumerate(clip[use_key].T):
            std_err = np.nanstd(data)/np.sqrt(n)
            h = std_err * t.ppf((1 + confidence) / 2, n - 1)
            clip_ave['disc_conf'][i]=h
            
    conf= [d[0] for d in clip_ave['disc_conf']]
    
    #bar plot with errorbars:
    ax.bar(xlabels, values, yerr=conf)
    # my=np.nanmax([v+conf[i] for i,v in enumerate(values)])
    # myy=np.nanmax(clip[use_key].flat)
    my=np.nanmax(values)
    not_isnan =  [not any(x) for x in np.isnan(clip[use_key])]
    nan_removed=clip[use_key][not_isnan,:]
    ax,h=connected_lines([0,1,2],nan_removed,ax=ax,color=color)
    
    #Perform paired t-test:
    _,p = ttest_rel(nan_removed[:,0],nan_removed[:,1])
    
    sig_star([0,1],p,my,ax)
    print(str(p))
    
    #Axis labels (here or outside of this to be more generalized)

    if axflag == False:
        return fig, ax, h
    else:
        return ax, h
    
def trial_part_position(raw,meta,ax=None,highlight=0,hl_color='b'):
    '''
    

    Parameters
    ----------
    raw : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    ax : Array of n=3 Matplotlib axis handles, optional (new figure made without)
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    axflag = True
    if ax == None:
        fig,ax = plt.subplots(1,3)
        axflag=False #No axis handle passed in    
    arena_size=23
    if highlight == 1:
        #Highlight Zone 1
        ax[1].fill_between([-arena_size,0],
                           [-arena_size,-arena_size],
                           [arena_size,arena_size],
                           hl_color, alpha=0.3,edgecolor='none')
    elif highlight ==2:
        #Highlight Zone 2
        ax[1].fill_between([0,arena_size],
                           [-arena_size,-arena_size],
                           [arena_size, arena_size],
                           hl_color, alpha=0.3,edgecolor='none')
        
    xx,yy=behavior.trial_part_position(raw,meta)
    for x,y,a in zip(xx,yy,ax):
        a.scatter(x,y,2,'k',marker='.',alpha=0.05)
        a.plot([0,0],[-22,22],'--r')
        # a.axis('equal')
        a.set_xlim([-22, 22])
        plt.sca(a)
        plt.xlabel('cm')
        plt.ylabel('cm')
    meta=behavior.stim_xy_loc(raw,meta)
    for x,y in zip(meta['stim_on_x'],meta['stim_on_y']):
        ax[1].plot(x,y,'bo')
        
    if axflag == False:
        return fig, ax
    else:
        return ax
    
def plot_openloop_day(raw,meta):    
    '''
    

    Parameters
    ----------
    raw : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    #### Set up figure axes in desired pattern
    fig = plt.figure(constrained_layout = True,figsize=(8.5,11))
    gs = fig.add_gridspec(6, 3)
    f_row=list(range(gs.nrows))
    f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
    f_row[1]=[fig.add_subplot(gs[1,0:2]) , fig.add_subplot(gs[1,2])]
    f_row[2]=[fig.add_subplot(gs[2,0:2]) , fig.add_subplot(gs[2,2])]
    f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
    f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
    f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]
    
    #### Row 0: Mouse position over trial parts 
    ax_pos = trial_part_position(raw,meta,ax=f_row[0])
    plt.sca(ax_pos[1])
    plt.title('%s, %s   %s   %s %s %s' % (meta['anid'][0],
                          meta['etho_exp_date'][0],
                          meta['protocol'][0],
                          meta['cell_type'][0],
                          meta['opsin_type'][0],
                          meta['stim_area'][0]))
    

    #### Calculate stim-triggered speed changes:
    baseline= round(np.mean(meta['stim_dur']))
    stim_dur= baseline
    vel_clip=behavior.stim_clip_grab(raw,meta,y_col='vel',
                                     stim_dur=stim_dur)
    
    clip_ave=behavior.stim_clip_average(vel_clip)
    
    #### Calculate stim-triggered %time mobile:
    percentage = lambda x: (np.nansum(x)/len(x))*100
    raw['m']=~raw['im']
    m_clip=behavior.stim_clip_grab(raw,meta,y_col='m', 
                                   stim_dur=stim_dur,
                                   summarization_fun=percentage)

    #### Row 1: Speed related
    ax_speedbar = mean_cont_plus_conf(clip_ave,
                                      xlim=[-stim_dur,stim_dur*2],
                                      highlight=[0,stim_dur,25],
                                      ax=f_row[1][0])
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    
    ax_speed = mean_bar_plus_conf(vel_clip,['Pre','Dur','Post'],ax=f_row[1][1])
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    
    #### Row 2: % Time mobile & (Rearing?)
    ax_im = mean_bar_plus_conf(m_clip,['Pre','Dur','Post'],ax=f_row[2][0])
    plt.ylabel('% Time Mobile')
    
    min_bout=1
    amb_bouts=behavior.bout_analyze(raw,meta,'ambulation',
                                    stim_dur=stim_dur,
                                    min_bout_dur_s=min_bout)
    im_bouts=behavior.bout_analyze(raw,meta,'im',
                                   stim_dur=stim_dur,
                                   min_bout_dur_s=min_bout)
    
    
    #### Row 3: Ambulation bout info
    #Rate
    ax_amb_bout_rate= mean_bar_plus_conf(amb_bouts,['Pre','Dur','Post'],
                                               use_key='rate',ax=f_row[3][0])
    plt.ylabel('Amb. bouts / 30s')
    
    #Duration
    ax_amb_bout_dur = mean_bar_plus_conf(amb_bouts,['Pre','Dur','Post'],
                                               use_key='dur',ax=f_row[3][1])
    plt.ylabel('Amb. dur (s)')
    
    #Speed
    ax_amb_bout_speed= mean_bar_plus_conf(amb_bouts,['Pre','Dur','Post'],
                                               use_key='speed',ax=f_row[3][2])
    plt.ylabel('Amb. speed (cm/s)')
    
    #### Row 4: immobility bout info
    #Rate
    ax_im_bout_rate= mean_bar_plus_conf(im_bouts,['Pre','Dur','Post'],
                                               use_key='rate',ax=f_row[4][0])
    plt.ylabel('Im. bouts / 30s')
    
    #Duration
    ax_im_bout_dur= mean_bar_plus_conf(im_bouts,['Pre','Dur','Post'],
                                               use_key='dur',ax=f_row[4][1])
    plt.ylabel('Im. dur (s)')
    
    ax_im_bout_speed= mean_bar_plus_conf(im_bouts,['Pre','Dur','Post'],
                                               use_key='speed',ax=f_row[4][2])
    plt.ylabel('Im. speed (cm/s)')
    
    #### Row 5: Meander/directedness (in progress)
    #Amb meander 
    ax_amb_meander= mean_bar_plus_conf(amb_bouts,
                                             ['Pre','Dur','Post'],
                                               use_key='meander',
                                               ax=f_row[5][0])
    plt.ylabel('Ambulation meander (deg/cm)')
    
    #All meander 
    #raw['meander']= behavior.measure_meander(raw,meta,use_dlc=False)
    # meander_clip=behavior.stim_clip_grab(raw,meta,y_col='meander',
    #                                      stim_dur=stim_dur,
    #                                      summarization_fun = np.nanmedian)
    # ax_all_meander= mean_bar_plus_conf(meander_clip,['Pre','Dur','Post'],
    #                                            ax=f_row[5][1])
    # plt.ylabel('Meadian meander (deg/cm)')
    
    raw['directed']= 1/ raw['meander']
    meander_clip=behavior.stim_clip_grab(raw,meta,y_col='directed',
                                         stim_dur=stim_dur,
                                         summarization_fun = np.nanmedian)
    ax_all_direct= mean_bar_plus_conf(meander_clip,['Pre','Dur','Post'],
                                               ax=f_row[5][2])
    plt.ylabel('Directed (cm/deg)')
    return fig

def plot_zone_day(raw,meta):    
    '''
    

    Parameters
    ----------
    raw : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    #### Set up figure axes in desired pattern
    fig = plt.figure(constrained_layout = True,figsize=(8.5,11))
    gs = fig.add_gridspec(6, 3)
    f_row=list(range(gs.nrows))
    f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)] #Position
    f_row[1]=[fig.add_subplot(gs[1,0:3])] #% Time spent pre,dur,post
    f_row[2]=[fig.add_subplot(gs[2,0:2]) , fig.add_subplot(gs[2,2])]
    f_row[3]=[fig.add_subplot(gs[3,0:2]) , fig.add_subplot(gs[3,2])]
    f_row[4]=[fig.add_subplot(gs[4,0:2]) , fig.add_subplot(gs[4,2])]
    f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]
    
    
    #### Title
    plt.sca(f_row[0][1])
    plt.title('%s, %s   %s   %s %s %s' % (meta['anid'][0],
                      meta['etho_exp_date'][0],
                      meta['protocol'][0],
                      meta['cell_type'][0],
                      meta['opsin_type'][0],
                      meta['stim_area'][0]));
    
    #### Row 0: Mouse position over trial parts 
    zone = int(meta['zone'][0].split(' ')[-1])
    ax_pos = trial_part_position(raw,meta,ax=f_row[0],highlight=zone)
    

    #### Row 1: %Time in zone 1 and 2:
    percentage = lambda x: (np.nansum(x)/len(x))*100
      
    t=[i for i in range(4)]
    t[0]=0
    t[1]=meta.task_start[0]
    t[2]=meta.task_stop[0]
    t[3]=meta.exp_end[0]
    in_zone1=[]
    for i in range(len(t)-1):
        ind=(raw['time']>= t[i]) & (raw['time'] < t[i+1])
        in_zone1.append(percentage(np.array(raw['iz1'].astype(int))[ind]))
        
    in_zone2=np.array([100,100,100])-np.array(in_zone1)
    x = np.array([0,1,2])
    width=0.35
    labels=['Pre','Dur', 'Post']
    f_row[1][0].bar(x - width/2,in_zone1,width,label='Zone 1')
    f_row[1][0].bar(x + width/2,in_zone2,width,label='Zone 2')
    f_row[1][0].set_ylabel('%Time in Zone')
    f_row[1][0].set_xticks(x)
    f_row[1][0].set_xticklabels(labels)
    f_row[1][0].legend()
    f_row[1][0].set_ylim([0,100])
    f_row[1][0].plot([-3,5],[50,50],'--k')
    f_row[1][0].set_xlim([-0.5, 3])
    
    ### Calculate stim-triggered speed changes:
    baseline= round(np.mean(meta['stim_dur']))
    stim_dur= baseline
    vel_clip=behavior.stim_clip_grab(raw,meta,y_col='vel',
                                      stim_dur=stim_dur)
    
    clip_ave=behavior.stim_clip_average(vel_clip)
    
    #### Row 2: Speed related
    ax_speedbar = mean_cont_plus_conf(clip_ave,
                                      xlim=[-stim_dur,stim_dur*2],
                                      highlight=[0,stim_dur,25],
                                      ax=f_row[2][0])
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    
    ax_speed = mean_bar_plus_conf(vel_clip,['Pre','Dur','Post'],
                                  ax=f_row[2][1],
                                  color='k')
    
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    
    
    #### Row 3: %time in SZ vs. time normalized to baseline
    width=5
    bin_size= width *60 
    st=0
    pbins=[]
    x=[]
    if zone == 1:
        sz = 'iz1'
        nsz = 'iz2'
    else:
        sz = 'iz2'
        nsz = 'iz1'
    while (st + bin_size) < t[3]:        
        start=st
        st=start + bin_size
        x.append(st)
        ind=(raw['time']>= start) & (raw['time'] < st)
        pbins.append(percentage(np.array(raw[sz].astype(int))[ind]))

    x = (np.array(x) - t[1])/60
    pbins=np.array(pbins)
    baseline= np.mean(pbins[x<0])
    pbins=pbins/baseline
    hl_color = 'b'
    f_row[3][0].fill_between([0,(t[2]-t[1])/60],[2,2],[0,0],
                        hl_color, alpha=0.3,edgecolor='none')
    f_row[3][0].bar(x - width/2,pbins,width-0.5,facecolor='k')
    f_row[3][0].set_ylabel('Norm. %Time SZ')
    f_row[3][0].set_xlabel('Time (m)')

    f_row[3][0].set_xlim((x[0] - width, x[-1]))
    f_row[3][0].set_ylim((0,2))
    f_row[3][0].plot([x[0]-width,x[-1]],[1,1],'--k')
    
    #Norm SZ time 3 bars Pre Dur Post
    in_sz=[]
    for i in range(len(t)-1):
        ind=(raw['time']>= t[i]) & (raw['time'] < t[i+1])
        in_sz.append(percentage(np.array(raw[sz].astype(int))[ind]))
        
    in_sz=np.array(in_sz)/in_sz[0]
    x = np.array([0,1,2])
    width=0.8
    labels=['Pre','Dur', 'Post']
    f_row[3][1].bar(x ,in_sz,width,facecolor='k')
    f_row[3][1].set_xticks(x)
    f_row[3][1].set_ylabel('Norm. %Time SZ')
    f_row[3][1].set_xticklabels(labels)
    f_row[3][1].set_ylim([0,2])
    
    #### Row 4: SZ speed vs. time normalized to baseline
    width=5
    bin_size= width *60 
    st=0
    pbins=[]
    x=[]
    while (st + bin_size) < t[3]:        
        start=st
        st=start + bin_size
        x.append(st)
        ind=((raw['time']>= start) & (raw['time'] < st)) & raw[sz]
        pbins.append(np.median(np.array(raw['vel'].astype(float))[ind]))

    x = (np.array(x) - t[1])/60
    pbins=np.array(pbins)
    baseline= np.mean(pbins[x<0])
    pbins=pbins/baseline
    hl_color = 'b'
    f_row[4][0].fill_between([0,(t[2]-t[1])/60],[2,2],[0,0],
                        hl_color, alpha=0.3,edgecolor='none')
    f_row[4][0].bar(x - width/2,pbins,width-0.5,facecolor='k')
    f_row[4][0].set_ylabel('Norm. SZ Speed')
    f_row[4][0].set_xlabel('Time (m)')

    f_row[4][0].set_xlim((x[0] - width, x[-1]))
    f_row[4][0].set_ylim((0,4))
    f_row[4][0].plot([x[0]-width,x[-1]],[1,1],'--k')
    
    #Norm SZ speed 3 bars Pre Dur Post
    in_sz=[]
    for i in range(len(t)-1):
        ind=((raw['time']>= t[i]) & (raw['time'] < t[i+1])) & raw[sz]
        in_sz.append(np.median(np.array(raw['vel'].astype(float))[ind]))
        
    in_sz=np.array(in_sz)/in_sz[0]
    x = np.array([0,1,2])
    width=0.8
    labels=['Pre','Dur', 'Post']
    f_row[4][1].bar(x ,in_sz,width,facecolor='k')
    f_row[4][1].set_xticks(x)
    f_row[4][1].set_ylabel('Norm. SZ Speed')
    f_row[4][1].set_xticklabels(labels)
    f_row[4][1].set_ylim([0,2])
    
    return fig

def zone_day_crossing_stats(raw,meta):
    
    ac_on,ac_off= signal.thresh(raw['iz1'].astype(int),0.5,'Pos')
    min=meta['fs'][0] * 4 #4 seconds
    all_cross=[]
    for on,off in zip(ac_on,ac_off):
        if (off-on) > min:
            all_cross.append([on,off])
    print('%d crossings detected.' % len(all_cross))
    new_meta=meta
    baseline=4 # seconds before / after crossing
    for i,cross in enumerate(all_cross):
        new_meta.loc[i,'stim_on']=raw['time'][cross[0]]
        new_meta.loc[i,'stim_off']=raw['time'][cross[0]]
    fs=meta['fs'][0]
    on_time =raw['time'][np.array(all_cross)[:,0]].values
    vel_cross=behavior.stim_clip_grab(raw,new_meta,'vel',x_col='time',
                            stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)
    x_cross=behavior.stim_clip_grab(raw,new_meta,'x',x_col='time',
                            stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)
    y_cross=behavior.stim_clip_grab(raw,new_meta,'y',x_col='time',
                            stim_dur=0,baseline=baseline,summarization_fun=np.nanmean)
    
    t=[i for i in range(4)]
    t[0]=0
    t[1]=meta.task_start[0]
    t[2]=meta.task_stop[0]
    t[3]=meta.exp_end[0]
    
    fig = plt.figure(constrained_layout = True,figsize=(9.48, 7.98))
    col_labs=['Pre','Dur','Post']
    gs = fig.add_gridspec(6, 3)
    f_row=list(range(gs.nrows))
    f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
    f_row[1]=[fig.add_subplot(gs[1,i]) for i in range(3)]
    f_row[2]=[fig.add_subplot(gs[2,i]) for i in range(3)]
    f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
    f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
    f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]
    
    
    #### Row 0: Plot trajectories overlayed
    r=0
    xtick_samps=np.array([x for x in range(0,x_cross['cont_y'].shape[0]+1,round(fs*2))])
    xticklab=[str(round(x)) for x in (xtick_samps/fs)-baseline]
    zero_samp=int(xtick_samps[-1]/2)
    for i,a in enumerate(f_row[r]):
        ind = (on_time > t[i]) & (on_time < t[i+1])
        cross_x=x_cross['cont_y'][:,ind]
        cross_y=y_cross['cont_y'][:,ind]
        for x,y in zip(cross_x.T,cross_y.T):
            b=y[zero_samp]
            y= y-b
            a.scatter(-x,y,2,alpha=0.05,facecolor='k',)
            
        a.plot([0,0],[-25,25],'--r')
        a.set_xlabel('Dist (cm)')
        a.set_xlim([-25,25])
        a.set_title(col_labs[i])
        
    ####Row 1: histogram of X on crossings:
    dist_bins=np.array([x for x in range(-22,22+1,2)])
    ncross_tot=vel_cross['cont_y'].shape[1]
    
    width=1.5
    r=1
    for i,a in enumerate(f_row[r]):
        ind = (on_time > t[i]) & (on_time < t[i+1])
        cross_x=x_cross['cont_y'][:,ind]
        ncross=sum(ind)
        dist_mat=np.ones((ncross,len(dist_bins)-1))
        ii=0
        for dist in cross_x.T:
            dist_mat[i,:],_ = np.histogram(-dist,dist_bins)
            ii += 1
            tot_hist=np.nansum(dist_mat,axis=0)
            tot_hist = 100*(tot_hist/np.nansum(tot_hist))
        a.bar(dist_bins[0:-1] - width/2,tot_hist,width,)    
        a.set_ylabel('% of 8s Crossing')
        a.set_xlabel('Dist (cm)')
        a.set_ylim([0,10])
    
    
    
    ####Row 2 vel clips full colorscale:
    r=2
    xtick_samps=[x for x in range(0,vel_cross['cont_y'].shape[0]+1,round(fs*2))]
    xticklab=[str(round(x)) for x in (xtick_samps/fs)-baseline]
    
    for i,a in enumerate(f_row[r]):
        ind = (on_time > t[i]) & (on_time < t[i+1])
        a.imshow(vel_cross['cont_y'][:,ind].T,
                 aspect='auto')
        plt.sca(a)
        plt.xticks(xtick_samps,xticklab)
        yticks=[x for x in range (0,sum(ind),5)]
        yticklab=[str(y) for y in yticks]
        plt.yticks(yticks,yticklab)
        a.set_xlabel('Time (s)')
        a.set_ylabel('Cross #')
        
        
    ####Row 3: Plot beginning, middle, and late crossing mean velocity:
    labels=['Early','Middle','Late']
    keep=np.ones((xtick_samps[-1],3))
    r=3
    for i,a in enumerate(f_row[r]):
        ind = (on_time > t[i]) & (on_time < t[i+1])
        cross_period=vel_cross['cont_y'][:,ind]
        ncross=sum(ind)
        step=round(ncross/3)
        eml =[x for x in range(0,ncross,step)]
        eml = eml[0:3]
        for ii,j in enumerate(eml):
            vel_stage=cross_period[:,j:(j+step)]
            x=vel_cross['cont_x']
            y=np.mean(vel_stage,axis=1)
            # if i > 0:
            #      y = y - baseline_change
            if i==0:
                keep[:,ii]=y
            a.plot(x,y,label=labels[ii])
            a.set_ylim([0,30])
        if i ==0:
            baseline_change=np.mean(keep,axis=1)
            # a.plot(x,np.mean(keep,axis=1),'--k')
        if i == 2:    
            a.legend()
            
        a.set_xlabel('Time (s)')
        a.set_ylabel('Speed (cm/s)')
    
    #### Row 4: Speed vs. distance from crossing
    r=4
    # Bin velocity of cross by x-distance of cross:
    dist_bins=[x for x in range(-16,16+1,2)]
    ncross_tot=vel_cross['cont_y'].shape[1]
    dist_mat=np.ones((ncross_tot,len(dist_bins)))
    i=0
    for dist,vel in zip(x_cross['cont_y'].T,vel_cross['cont_y'].T):
        ind = np.digitize(-dist,dist_bins)
        for x in range(np.min(ind),np.max(ind)):
            use= ind == x
            if any(use):
                dist_mat[i,x]=np.median(vel[use])
            else:
                dist_mat[i,x]=np.nan
        i += 1
    xtick_samps=np.array([x for x in range(0,len(dist_bins),1)])
    xticklab=[str(round(x)) for x in dist_bins]
    
    for i,a in enumerate(f_row[r]):
        ind = (on_time > t[i]) & (on_time < t[i+1])
        a.imshow(dist_mat[ind,:], aspect='auto')
        plt.sca(a)
        plt.xticks(xtick_samps,xticklab)
        yticks=[x for x in range (0,sum(ind),5)]
        yticklab=[str(y) for y in yticks]
        plt.yticks(yticks,yticklab)
        a.set_xlabel('Dist from SZ (cm)')
        a.set_ylabel('Cross #')
    
    #### Row 5: Plot beginning, middle, and late crossing mean velocity:
    labels=['Early','Middle','Late']
    r=5
    keep=np.ones((len(xtick_samps),3))
    for i,a in enumerate(f_row[r]):
        ind = (on_time > t[i]) & (on_time < t[i+1])
        cross_period=dist_mat[ind,:]
        ncross=sum(ind)
        step=round(ncross/3)
        eml =[x for x in range(0,ncross,step)]
        eml = eml[0:3]
        for ii,j in enumerate(eml):
            vel_stage=cross_period[j:(j+step),:]
            x=dist_bins
            y=np.nanmean(vel_stage,axis=0)
            if i==0:
                keep[:,ii]=y
            a.plot(x,y,label=labels[ii])
            a.set_ylim([0,30])
        # if i ==1:
            # a.plot(x,np.mean(keep,axis=1),'--k')
        if i == 2:    
            a.legend()
            
        a.set_xlabel('Dist from SZ (cm)')
        a.set_ylabel('Speed (cm/s)')
    return fig

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