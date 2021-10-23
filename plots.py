#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:37:06 2020

@author: brian
"""
# import cv2
import os
import seaborn as sns
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt

import numpy as np
from matplotlib.pyplot import gca 
from matplotlib.collections import PatchCollection
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec


from gittislab import table_wrappers
from gittislab import signals
from gittislab import behavior
from gittislab import dataloc
from gittislab import ethovision_tools
from gittislab import model

from scipy import stats as scistats
import statistics as pystats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

# from scipy.stats import ttest_rel
# from scipy.stats import t
import pdb 
from pathlib import Path
import pandas as pd 
import scipy

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
        h0, =ax.plot(x,row, color=color, linewidth=0.25)
        ax.plot(x,row,'ok', markersize= 1)
        h.append(h0)
    
    if axflag == False:
        return fig, ax, h
    else:
        return ax,h
    
def mean_cont_plus_conf_clip(clip_ave,
                             xlim=[-45,60],
                             ylim=[0, 20],
                             highlight=None,
                             hl_color='b',
                             ax=None):
    '''
    Parameters
    ----------
    clip_ave : DICT with fields:
        'cont_y' Array of continous data samples (mean of trace)
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
    plot_each: Boolean
        plot on each individual trace in addition to mean +/- conf. int.
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

    # pdb.set_trace()
    #If highlight patch box specified, plot it first:
    if highlight:
        ax.fill_between(highlight[0:2],[highlight[2],highlight[2]],y2=[0,0],
                        color=hl_color, alpha=0.3,edgecolor='none')
        
        
    m,conf_int=clip_ave['cont_y_conf']
    ub = conf_int[:,0]  #Positive confidence interval
    lb = conf_int[:,1]  #Negative confidence interval
    ax.fill_between(x, ub, lb, color='k', alpha=0.3, edgecolor='none',
                    zorder=5)
    ax.plot(x,y,'k')
    plt.xticks(np.arange(-60,90,10))
    plt.yticks(np.arange(-0,20,5))
    plt.xlim(xlim)
    
    #Set major and minor tick labels
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    if xlim[1] >= 30:
        ax.xaxis.set_major_locator(MultipleLocator(30))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    plt.ylim(ylim)
    
    if ax == None:
        return fig,ax
    else:
        return ax

def mean_cont_plus_conf_array(x,
                              ymat,
                              ax=None,
                              confidence = 0.95,
                              plot_each=False,
                              highlight=None,
                              hl_color='b'):
    '''
      
    Parameters
    ----------       
    'x' Array of x-values matching rows in 'ymat'
    'y_mat' Array of continous data (rows=samples,cols=trials) 
    confidence: float,
            Represents confidence bounds to plot. e.g. 0.95 for 95% conf.
    highlight : LIST, optional
        3-value list describing rectangular patch to highlight on plot.
            [start x, stop x, height y]. The default is None. (No patch)
            E.g. [0,30,15] -- plot a rectangle from 0 to 30s, 15 high
    hl_color  : STR, optional
        string specifying color, default = 'b' (blue)
    plot_each: Boolean
        plot on each individual trace in addition to mean +/- conf. int.
    ax : matplotlib.pyplot.subplot axis handle if plotting in subplot
    Returns
    -------
    fig : matplotlib.pyplot.subplot figure handle
        DESCRIPTION.
    ax : matplotlib.pyplot.subplot axis handle
        DESCRIPTION.

       
    '''
    if ax == None: #If no axis provided, create a new plot
        fig,ax=plt.subplots()
    plt.sca(ax)

    #If highlight patch box specified, plot it first:
    if highlight:
        ax.fill_between(highlight[0:2],[highlight[2],highlight[2]],y2=[0,0],
                        color=hl_color, alpha=0.3,edgecolor='none')
    
    if plot_each:
        for y in ymat.T:
            plt.plot(x,y,'k')
            
    m, conf_int=signals.conf_int_on_matrix(ymat)
    ub = conf_int[:,0]  #Positive confidence interval
    lb = conf_int[:,1]  #Negative confidence interval
    ax.fill_between(x, ub, y2=lb, color='k', alpha=0.3, edgecolor='none',
                    zorder=5)
    
    #Plot mean on top:
    mean_color = 'k'
    if plot_each:
        mean_color = 'r'
    
    ax.plot(x,m,mean_color)
    

    
    if ax == None:
        return fig,ax
    else:
        return ax
    
def mean_bar_plus_conf_clip(clip,xlabels,use_key='disc',
                            ax=None,
                            clip_method=True,
                            horz = False,
                            bar_color= np.array([57,83,163])/255,
                            line_color=np.array([179,179,179])/255):
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
            h = std_err * scistats.t.ppf((1 + confidence) / 2, n - 1)
            clip_ave['disc_conf'][i]=h
            
    conf= [d[0] for d in clip_ave['disc_conf']]
    
    #bar plot with errorbars:
    if horz is False:
        bar_cont=ax.bar(xlabels, values,color=bar_color)

        # pdb.set_trace()
        # my=np.nanmax([v+conf[i] for i,v in enumerate(values)])
        # myy=np.nanmax(clip[use_key].flat)
        my=np.nanmax(values)
        not_isnan =  [not any(x) for x in np.isnan(clip[use_key])]
        nan_removed=clip[use_key][not_isnan,:]
        ax,h=connected_lines([0,1,2],nan_removed,ax=ax,color=line_color)
        
        #Plot conf intervals (gray ci95 by default)
        (_, caps, _) = plt.errorbar([-0.25,1.25,2.25],values,
                                    yerr=conf, 
                                    capsize=2,
                                    linestyle='None',
                                    ecolor='k')
        #Perform paired t-test:
        _,p = scistats.ttest_rel(nan_removed[:,0],nan_removed[:,1])
        
        sig_star([0,1],p,my,ax)
    elif horz == 'Left':
        # pdb.set_trace()
        ax.barh(xlabels, -values,color=bar_color)
        (_, caps, _) = plt.errorbar(-values,[-0.25,1.25,2.25],
                                    xerr=conf, 
                                    capsize=2,
                                    linestyle='None',
                                    ecolor='k')
    elif horz == 'Right':
        ax.barh(xlabels, values,  color=bar_color) #xerr=conf,
        (_, caps, _) = plt.errorbar(values,[-0.25,1.25,2.25],
                                    xerr=conf, 
                                    capsize=2,
                                    linestyle='None',
                                    ecolor='k')
    # print(str(p))
    
    if horz:
        ax.set_xlim([-1,1])
        ax.invert_yaxis()
    
    #Axis labels (here or outside of this to be more generalized)

    if axflag == False:
        return fig, ax, h
    else:
        return ax, h
    
def mean_bar_plus_conf_array(data,xlabels,ax=None,
                             color='k',
                             confidence = 0.95,
                             paired=True):
    '''
    Calc mean of data columns as bar plots +/- conf. 
    Create figure /axis and return handles if no axis passed.

    Parameters
    ----------
    data: np.array
        n x m array of data where m columns correspond to xlabels
    xlabels : LIST
        List of strings to label x axis of bar plot
    color (opt): string
        Color to use for connected line plots, default is black.
    ax : matplotlib axis handle, optional
        Axis handle of which subplot to use (if provided). The default is None,
        which will create a new figure when function is called and return new 
        figure handle and axis handles.
    confidence: float, optional
        Fractional confidence interval to plot as error bars (default 0.95 = 95% CI)
    
    Returns
    -------
    fig = figure handle (if a figure was generated (i.e. ax =None))
    ax = axis handle of barplot axis
    h = list of connected line handles

    '''
    #Barplot +/- conf
    
    axflag = True
    if ax == None:
        fig,ax = plt.subplots()
        axflag=False
    plt.sca(ax)
    conf=[]
    if paired == True:
        values=np.nanmean(data,axis=0)
        n=data.shape[0]        

        for i,col in enumerate(data.T):
            std_err = np.nanstd(col)/np.sqrt(n)
            ci = std_err * scistats.t.ppf((1 + confidence) / 2, n - 1)
            conf.append(ci)
                
        not_isnan =  [not any(x) for x in np.isnan(data)]
        nan_removed=data[not_isnan,:]
        
        #Perform paired t-test:
        _,p = scistats.ttest_rel(nan_removed[:,0],nan_removed[:,1])
    else:
        values=[]
        data_clean=[]
        for col in data:
            col=np.array(col)
            n=len(col)
            values.append(np.nanmean(col))
            std_err = np.nanstd(col)/np.sqrt(n)
            ci = std_err * scistats.t.ppf((1 + confidence) / 2, n - 1)
            conf.append(ci)            
            data_clean.append(col[~np.isnan(col)])
        # pdb.set_trace()
        _,p= scistats.ttest_ind(data_clean[0],data_clean[1])
        h=[]
    #bar plot with errorbars:
    ax.bar(xlabels, values, yerr=conf)
    my=np.nanmax(values)

    if paired == True:        
        x_connect=[i for i in range(0,len(xlabels))]        
        ax,h=connected_lines(x_connect,nan_removed,ax=ax,color=color)
        
    sig_star([0,1],p,my *1.1,ax)
    print(str(p))
    
    #Axis labels (here or outside of this to be more generalized)

    if axflag == False:
        return fig, ax, h
    else:
        return ax, h 
    
def trial_part_position(raw,meta,ax=None,highlight=0,hl_color='b',chunk_method='task'):
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

        
    xx,yy=behavior.trial_part_position(raw,meta,
                                       chunk_method=chunk_method)
    arena_size=23
    
    pre_dur_post_arena_plotter(xx,yy,ax,highlight=highlight,color=hl_color,
                               arena_size=arena_size)
    
    meta=behavior.stim_xy_loc(raw,meta)
    for x,y in zip(meta['stim_on_x'],meta['stim_on_y']):
        ax[1].plot(x,y,'bo')
        
    if axflag == False:
        return fig, ax
    else:
        return ax
    
def pre_dur_post_arena_plotter(xx,yy,ax,highlight=0,color='b',arena_size=23):
    
    if highlight == 1:
        #Highlight Zone 1
        ax[1].fill_between([-arena_size,0],
                           [-arena_size,-arena_size],
                           y2=[arena_size,arena_size],
                           facecolor = color, alpha=0.3,edgecolor='none')
    elif highlight ==2:
        #Highlight Zone 2
        ax[1].fill_between([0,arena_size],
                           [-arena_size,-arena_size],
                           y2=[arena_size, arena_size],
                           facecolor=color, alpha=0.3,edgecolor='none')
    for x,y,a in zip(xx,yy,ax):
        a.scatter(x,y,10,'k',marker='.',alpha=0.05,rasterized = True)
        a.plot([0,0],[-arena_size, arena_size],'--r')
        a.set_xlim([-arena_size, arena_size])
        plt.sca(a)
        plt.xlabel('cm')
        plt.ylabel('cm')    
        
def plot_openloop_day(raw,meta,save=False, close=False,save_dir=None):    
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
    if meta['no_trial_structure'][0] == False:
        #### Set up figure axes in desired pattern
        plt_ver = 3
        fig = plt.figure(constrained_layout = False,figsize=(8.5,11))
        gs = fig.add_gridspec(6, 3)
        f_row=list(range(gs.get_geometry()[0]))
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
    
        #### Row 0: Speed related
        ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                          xlim=[-stim_dur,stim_dur*2],
                                          highlight=[0,stim_dur,25],
                                          ax=f_row[1][0])
        plt.ylabel('Speed (cm/s)')
        plt.xlabel('Time from stim (s)')
        
        ax_speed = mean_bar_plus_conf_clip(vel_clip,['Pre','Dur','Post'],ax=f_row[1][1])
        plt.ylabel('Speed (cm/s)')
        plt.xlabel('Time from stim (s)')
        
        #### Row 1: % Time mobile & (Rearing?)
        ax_im = mean_bar_plus_conf_clip(m_clip,['Pre','Dur','Post'],ax=f_row[2][0])
        plt.ylabel('% Time Mobile')
        
        min_bout=1
        amb_bouts=behavior.bout_analyze(raw,meta,'amb',
                                        stim_dur=stim_dur,
                                        min_bout_dur_s=min_bout)
        im_bouts=behavior.bout_analyze(raw,meta,'im',
                                       stim_dur=stim_dur,
                                       min_bout_dur_s=min_bout)
        # pdb.set_trace()
        #Rearing:
        if meta['has_dlc'][0] == True:
            rate = lambda x: len(signals.thresh(x,0.5)[0]) / stim_dur
            rear_clip=behavior.stim_clip_grab(raw,meta,y_col='rear', 
                                   stim_dur=stim_dur,
                                   summarization_fun=rate)
            mean_bar_plus_conf_clip(rear_clip,['Pre','Dur','Post'],ax=f_row[2][1])
            plt.ylabel('Rears / s (Hz)')
    
        #### Row 2: Ambulation bout info
        #Rate
        ax_amb_bout_rate= mean_bar_plus_conf_clip(amb_bouts,['Pre','Dur','Post'],
                                                   use_key='rate',ax=f_row[3][0])
        plt.ylabel('Amb. bouts / 30s')
        
        #Duration
        ax_amb_bout_dur = mean_bar_plus_conf_clip(amb_bouts,['Pre','Dur','Post'],
                                                   use_key='dur',ax=f_row[3][1])
        plt.ylabel('Amb. dur (s)')
        
        #Speed
        ax_amb_bout_speed= mean_bar_plus_conf_clip(amb_bouts,['Pre','Dur','Post'],
                                                   use_key='speed',ax=f_row[3][2])
        plt.ylabel('Amb. speed (cm/s)')
        
        #### Row 3: immobility bout info
        #Rate
        ax_im_bout_rate= mean_bar_plus_conf_clip(im_bouts,['Pre','Dur','Post'],
                                                   use_key='rate',ax=f_row[4][0])
        plt.ylabel('Im. bouts / 30s')
        
        #Duration
        ax_im_bout_dur= mean_bar_plus_conf_clip(im_bouts,['Pre','Dur','Post'],
                                                   use_key='dur',ax=f_row[4][1])
        plt.ylabel('Im. dur (s)')
        
        ax_im_bout_speed= mean_bar_plus_conf_clip(im_bouts,['Pre','Dur','Post'],
                                                   use_key='speed',ax=f_row[4][2])
        plt.ylabel('Im. speed (cm/s)')
        
        #### Row 4: Unilateral bias
        
        
        
        #Amb meander -- exclude for now
        # ax_amb_meander= mean_bar_plus_conf_clip(amb_bouts,
        #                                          ['Pre','Dur','Post'],
        #                                            use_key='meander',
        #                                            ax=f_row[5][0])
        # plt.ylabel('Ambulation meander (deg/cm)')
        
        #All meander 
        #raw['meander']= behavior.measure_meander(raw,meta,use_dlc=False)
        # meander_clip=behavior.stim_clip_grab(raw,meta,y_col='meander',
        #                                      stim_dur=stim_dur,
        #                                      summarization_fun = np.nanmedian)
        # ax_all_meander= mean_bar_plus_conf_clip(meander_clip,['Pre','Dur','Post'],
        #                                            ax=f_row[5][1])
        # plt.ylabel('Meadian meander (deg/cm)')
        
        # raw['directed']= 1/ raw['meander']
        # meander_clip=behavior.stim_clip_grab(raw,meta,y_col='directed',
        #                                      stim_dur=stim_dur,
        #                                      summarization_fun = np.nanmedian)
        # ax_all_direct= mean_bar_plus_conf_clip(meander_clip,['Pre','Dur','Post'],
        #                                            ax=f_row[5][2])
        # plt.ylabel('Directed (cm/deg)')
        
        #### Save image option:
        if save == True:
            if save_dir == None:
                path_dir = str(meta['pn'][0].parent)
            else:
                path_dir = str(save_dir)
            anid= meta['anid'][0]
            proto=meta['etho_exp_type'][0]
            plt.show(block=False)
            plt.savefig(path_dir + '/%s_%s_summary_v%d.png' %  (anid,'open_loop',plt_ver))
        if close == True:
            plt.close()
        else:
            return fig
    
def plot_freerunning_day(raw,meta,save=False, close=False):    
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
    #Check that these data are free-running type:
    if meta['no_trial_structure'][0] == True:
        #### Set up figure axes in desired pattern
        plt_ver = 3
        fig = plt.figure(constrained_layout = True,figsize=(10,7))
        gs = fig.add_gridspec(4, 3)
        f_row=list(range(gs.get_geometry()[0]))
        f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
        f_row[1]=[fig.add_subplot(gs[1,0:3])]
        f_row[2]=[fig.add_subplot(gs[2,0:2]) , fig.add_subplot(gs[2,2])]
        f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
        
        #### Row 0: Mouse position over 3 chunks
        ax_pos = trial_part_position(raw,meta,ax=f_row[0],
                                     chunk_method='thirds')
        plt.sca(ax_pos[1])
        plt.title('%s, %s %s (%s) %s %s %s' % (meta['anid'][0],
                              meta['etho_exp_date'][0],
                              meta['protocol'][0],
                              meta['etho_exp_type'][0],
                              meta['cell_type'][0],
                              meta['opsin_type'][0],
                              meta['stim_area'][0]))
        
        #Analyze in thirds by adding a pseudo stim-time 1/3 way through:
        #i.e. Pre Dur Post analysis code will analyze 1st, 2nd, and 3rd chunk of
        #free_running / unstructured open field data
        
        pseudo_stim = meta
        third=meta['exp_end'][0]/3
        pseudo_stim['stim_on'] = third
        pseudo_stim['stim_off'] = 2*third
        pseudo_stim['stim_dur']=third
        data=behavior.experiment_summary_helper(raw,
                                              pseudo_stim,
                                              min_bout=0.5)
        #### Row 1: Plot binned speed vs. time
        # vel=raw['vel']
        # vel[0:2]=np.nan
        # x,y = signals.bin_analyze(raw['time'],vel,
        #                           bin_dur=10,
        #                           fun = np.nanmedian)
        plt.sca(f_row[1][0])
        x = data['x_bin']
        
        plt.plot(x/60,data['vel_bin'],'k')
        plt.ylabel('Speed (cm/s)')
        plt.xlabel('Time (min)')
    
        
        #### Row 2 Left: % Time mobile vs. time
        plt.sca(f_row[2][0])
        # percentage = lambda x: (np.nansum(x)/len(x))*100        
        # mobile = ~raw['im']
        # x,y = signals.bin_analyze(raw['time'],mobile,
        #                   bin_dur=10,
        #                   fun = percentage)

        plt.plot(x/60,data['raw_per_mobile'],'k')
        plt.ylabel('% Time Mobile')
        plt.xlabel('Time (min)')
        
        #Row 2 Right: Proportion time spent in various behaviors:

        dat = data['prop_state']
        tot = [1,1,1]
        labs= data['prop_labels']
        width = 0.4       # the width of the bars: can also be len(x) sequence
        cols=['k','w','g']
        ax=f_row[2][1]
        labels=['%d-%d' % ((i-1)*third/60,i*third/60) for i in range(1,4)]
        # pdb.set_trace()
        for i,b in enumerate(labs):
            bt=[0,0,0]
            if i>0:
                bt=np.sum(dat[0:i,:],axis=0)
            ax.bar(labels, dat[i,:], width, label = b,
                   edgecolor='k',facecolor=cols[i],bottom=bt)
        ax.legend()
        ax.set_xlim([-1,5])
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Time (min)')
        
        #Row 3 Left: Ambulation speed
        ax=f_row[3][0]
        ax.bar(labels, data['amb_bout_rate'], width,
                   edgecolor='k',facecolor='w')
        ax.set_xlim([-1,3])
        ax.set_ylabel('Bout rate (Hz)')
        ax.set_xlabel('Time (min)')
        ax.set_title('Ambulation')
        
        #Row 3 Middle: Ambulation Bout Frequency
        ax=f_row[3][1]
        ax.bar(labels, data['amb_speed'], width,
                   edgecolor='k',facecolor='w')
        ax.set_xlim([-1,3])
        ax.set_ylabel('Speed cm/s')
        ax.set_xlabel('Time (min)')
        ax.set_title('Ambulation')
        
        #Row 3 Right: Immobility bout frequency
        ax=f_row[3][2]
        ax.bar(labels, data['im_bout_rate'], width,
                   edgecolor='k',facecolor='k')
        ax.set_xlim([-1,3])
        ax.set_ylabel('Bout rate (Hz)')
        ax.set_xlabel('Time (min)')
        ax.set_title('Immobility')
        
        
        #### Save image option:
        if save == True:
            path_dir = str(meta['pn'][0].parent)
            anid= meta['anid'][0]
            proto=meta['etho_exp_type'][0]
            plt.show(block=False)
            plt.savefig(path_dir + '/%s_%s_summary_v%d.png' %  (anid,proto,plt_ver))
        if close == True:
            plt.close()
        else:
            return fig
    else:
        desc_str='%s-%s-%s-%s-%s-%s' % (meta['anid'][0],
                              meta['etho_exp_date'][0],
                              meta['protocol'][0],
                              meta['cell_type'][0],
                              meta['opsin_type'][0],
                              meta['stim_area'][0])
        print('Skipping structured trial: %s' % desc_str)
        
def plot_freerunning_mouse_summary(data, save=False, close=False):    
    '''
    

    Parameters
    ----------
    data - pandas.DataFrame() output from function:
        gittislab.experiment_summary_collect()

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    
    #### Set up figure axes in desired pattern
    plt_ver = 1

    fig = plt.figure(constrained_layout = True,figsize=(10,7))
    gs = fig.add_gridspec(3, 3)
    f_row=list(range(gs.get_geometry()[0]))
    
    
    # Determine if there is a mixture of stimulation durations:
    durs = np.unique(data['stim_dur'])
    if len(durs) == 1:
        f_row[0]=[fig.add_subplot(gs[0,0:2]) , fig.add_subplot(gs[0,2])]
        sum_i=1
    elif len(durs)>=2:
        durs=durs[0:2]
        f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
        sum_i=2
        
    f_row[1]=[fig.add_subplot(gs[1,0:2]) , fig.add_subplot(gs[1,2])]
    f_row[2]=[fig.add_subplot(gs[2,0:2]) , fig.add_subplot(gs[2,2])]
    # f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
    # f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
    # f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]
    
    #### Row 0: Speed vs. time stim effects & bar summary 
    for i,dur in enumerate(durs):
        dat=data.loc[data['stim_dur']==dur,'vel_bin']
        cao=data['cell_area_opsin'][1]
        x=data.loc[data['stim_dur']==dur,'x_bin'].values
        y=np.vstack([x for x in dat])
        ym= np.mean(y,axis=0)
        clip_ave={'cont_y' : ym,
                  'cont_x' : x[0]/60,
                  'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
                  'disc' : np.vstack(data['amb_speed'].values)}
                  
        ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                          highlight=None,
                                          xlim=[0,(dur/60)*3],
                                          ax=f_row[0][i])
        plt.ylabel('Speed (cm/s)')
        plt.xlabel('Time from stim (s)')
        plt.title('%s %s, n=%d' % (cao,data['proto'][0],y.shape[0]))
    
    use_dur=pystats.mode(durs)
    labels=['{0:1.0f}-{1:1.0f}'.format((i-1)*use_dur/60,i*use_dur/60) 
            for i in range(1,4)]
    ax_speed = mean_bar_plus_conf_clip(clip_ave,labels,
                                  clip_method=False, ax=f_row[0][sum_i])
    plt.ylabel('Amb speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    plt.title('n=%d' % len(data))    

    #### Row 1 Left: %Time mobile
    plt.sca(f_row[1][0])
    dat=data.loc[:,'per_mobile']
    y=np.vstack([x for x in dat])
    ym= np.mean(y,axis=0)
    
    
    clip_ave={'cont_y' : ym,
              'cont_x' : [0, 1, 2],
              'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
              'disc' : y}
              
    ax_mobile= mean_bar_plus_conf_clip(clip_ave,labels,
                              clip_method=False, ax=f_row[1][0])
    plt.ylabel('Time Mobile (%)')
    plt.xlabel('Time from stim (s)')

    #### Row 1 Right: Proportion time spent doing various behaviors:
    dat = np.mean(np.stack(data['prop_state'],axis=0),axis=0)
    tot = [1,1,1]
    labs= data['prop_labels'][0]
    width = 0.4       # the width of the bars: can also be len(x) sequence
    cols=['k','w','g']
    ax=f_row[1][1]
    for i,b in enumerate(labs):
        bt=[0,0,0]
        if i>0:
            bt=np.sum(dat[0:i,:],axis=0)
        ax.bar(labels, dat[i,:], width, label = b,
               edgecolor='k',facecolor=cols[i],bottom=bt)
    ax.legend()
    ax.set_xlim([-1,5])
    ax.set_ylabel('Proportion')    
    
    #### Row 2: 0: Rate of rearing (?)
    
    plt.sca(f_row[0][0])
    
def plot_freerunning_cond_comparison(data,save=False,close=False):
    fig = plt.figure(constrained_layout = True,figsize=(10,7))
    gs = fig.add_gridspec(2, 5)
    f_row=list(range(gs.get_geometry()[0]))
    conds = [i for i in data.keys()]
    f_row[0]=[fig.add_subplot(gs[0,0:gs.ncols])]
    f_row[1]=[fig.add_subplot(gs[1,i]) for i in range(gs.ncols)]
   
    ax_speedbar=None
    ax=[]
    cols=['b','r']
    for cond in conds:            
            dat=data[cond].loc[:,'vel_bin'].values
            x=data[cond].loc[:,'x_bin'].values
            y=np.vstack([x for x in dat])
            ym= np.mean(y,axis=0)
            clip_ave={'cont_y' : ym,
                      'cont_x' : x[0]/60,
                      'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
                      'disc' : np.vstack(data[cond]['amb_speed'].values)}
            dur = 15
            ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                              highlight=None,
                                              xlim=[0,dur],
                                              ax=f_row[0][0])
    lines=ax_speedbar.get_lines()
    
    for cond,col,line in zip(conds,cols,lines):
        line.set_label(cond)
        line.set_color(col)
    plt.sca(ax_speedbar)
    ax_speedbar.set_ylabel('Speed (cm/s)')
    ax_speedbar.set_xlabel('Time (min)')
    plt.legend()
    
    if 'rear_bout_rate' in data[conds[0]].columns:
        bar_types=['per_mobile','amb_bout_rate','amb_speed','im_bout_rate',
                   'rear_bout_rate']
        ylabs=['Time mobile (%)','Amb bouts / s','Amb speed (cm/s)','Im bouts /s',
               'Rear bouts / s']
    else:        
        bar_types=['per_mobile','amb_bout_rate','amb_speed','im_bout_rate']
        ylabs=['Time mobile (%)','Amb bouts / s','Amb speed (cm/s)','Im bouts /s']
    label_columns=['0-5','5-10','10-15'] #Should be determined based on data thirds
    sns.set_theme(style="whitegrid")
    i=0
    # pdb.set_trace()
    for examine,ylab in zip(bar_types,ylabs):
        df = behavior.summary_collect_to_df(data,
                              use_columns=['anid',examine],
                              label_columns=label_columns,
                              var_name='time_window', 
                              value_name=examine,                        
                              static_columns=['anid'])
    
        ax = sns.barplot(ax=f_row[1][i],
                         x="time_window", 
                         y=examine,
                         hue='cond',
                         data=df,)
        if i < 4:
             ax.legend_.remove()
                
       
        ax.set_xlabel('Minutes')
        ax.set_ylabel(ylab)
        i += 1
        
def plot_openloop_mouse_summary(data, 
                                smooth_amnt= [33,33],
                                save=False,
                                close=False,
                                method = [10,1]):    
    '''
    

    Parameters
    ----------
    data - pandas.DataFrame() output from function:
        gittislab.experiment_summary_collect()

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
        
    stats: List
        Contains relevant statistics for each panel in the figure by row:
            [[col0, col1, col3], [col0,col12] , [etc]]

    '''
    #### Set up figure axes in desired pattern
    plt_ver = 3

    fig = plt.figure(constrained_layout = False,figsize=(8.5,11))
    gs = fig.add_gridspec(6, 3)
    f_row=list(range(gs.get_geometry()[0]))
    
    
    # Determine if there is a mixture of stimulation durations:
    durs = np.unique(data['stim_dur'])
    if method == 'each_dur':
        if (len(durs) == 1):
            f_row[0]=[fig.add_subplot(gs[0,0:2]) , fig.add_subplot(gs[0,2])]
            sum_i=1
        elif (len(durs)>=2):
            durs=durs[0:2]
            f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
            sum_i=2
    else: #Method is a list
        f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)]
        sum_i=2
        
    f_row[1]=[fig.add_subplot(gs[1,0:2]) , fig.add_subplot(gs[1,2])]
    f_row[2]=[fig.add_subplot(gs[2,i]) for i in range(3)]
    f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
    f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
    f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]
    
    stats=[[] for n in range(len(f_row))]
    
    #### Row 0: Speed vs. time stim effects & bar summary     
    # Note: add each dur as method 1, and method 2 will be first 9s stim & last 1s stim
    if method == 'each_dur':            
        for i,dur in enumerate(durs):
            dat=data.loc[data['stim_dur']==dur,'vel_trace'].values
            
            x=data.loc[data['stim_dur']==dur,'x_trace'].values
            
            #Smooth:
            #pdb.set_trace()
            y=np.vstack([signals.smooth(x,window_len=smooth_amnt[i],window='blackman')\
                         for x in dat])
            
            
            ym= np.nanmean(y,axis=0)
            clip_ave={'cont_y' : ym,
                      'cont_x' : x[0],
                      'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
                      'disc' : np.vstack(data['stim_speed'].values)}
                      
            ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                              xlim=[-dur,dur*2],
                                              ylim=[0, 10],
                                              highlight=[0,dur,25],
                                              ax=f_row[0][i],)
            plt.ylabel('Speed (cm/s)')
            plt.xlabel('Time from stim (s)')
            cao=data['cell_area_opsin'][1]
            plt.title('%s 10x%ds (%s), n=%d' % (cao,dur,data['proto'][0],y.shape[0]))
            rot_bias_dur=dur
            # mean_bar_plus_conf_clip(clip,xlabels,use_key='disc',ax=None,clip_method=True,
                           # color='')
    else: #Method is a list of how to make split axis plot
        daty = data.loc[:,'vel_trace'].values
        datDur = data.loc[:,'stim_dur'] 
        ys = [ [] , [] ]
    
        
        datx = data.loc[:,'x_trace'].values
        cao=data['cell_area_opsin'][1]
        t2 = method[1]
        t1 = method[0]-method[1]
        t0 = method[0]
        rot_bias_dur=t0
        for x,y,dur in zip(datx,daty,datDur):
            ind1 = (x >= -t0) &  (x < t1)
            endT = dur - t2
            ind2 = (x >= endT) & (x < (dur + t0))
            ys[0].append(y[ind1])
            ys[1].append(y[ind2])
            
        final_x1 = x[ind1]
        final_x2 = x[ind2] - dur + t0
        xx = [ final_x1, final_x2]
        for i, dat in enumerate(ys):
            lens = [len(x) for x in dat]
            m = np.min(lens)
            y=np.vstack([signals.smooth(x[0:m],window_len=smooth_amnt[i],window='blackman')\
                         for x in dat])
            
            
            ym= np.nanmean(y,axis=0)
            clip_ave={'cont_y' : ym,
                      'cont_x' : xx[i][0:m],
                      'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
                      'disc' : np.vstack(data['stim_speed'].values)}
                     
            ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                              xlim=[xx[i][0],xx[i][-1]],
                                              ylim=[0, 10],
                                              highlight=[0,dur,25],
                                              ax=f_row[0][i],)
            plt.ylabel('Speed (cm/s)')
            plt.xlabel('Time from stim (s)')
            plt.title('%s 10x%ds (%s), n=%d' % (cao,dur,data['proto'][0],y.shape[0]))
            
    ax_speed = mean_bar_plus_conf_clip(clip_ave,['Pre','Dur','Post'],
                                  clip_method=False, ax=f_row[0][sum_i],
                                  )
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    plt.title('n=%d' % len(data))    

    #### Row 1 Left: %Time mobile

    dat=data.loc[:,'per_mobile']
    y=np.vstack([x for x in dat])
    ym= np.mean(y,axis=0)
    
    
    clip_ave={'cont_y' : ym,
              'cont_x' : [0, 1, 2],
              'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
              'disc' : y}
              
    ax_mobile= mean_bar_plus_conf_clip(clip_ave,['Pre','Dur','Post'],
                              clip_method=False, ax=f_row[1][0], 
                              )
    plt.ylabel('Time Mobile (%)')
    plt.xlabel('Time from stim (s)')
    stats[1]+=[{'stat':0}]
    
    
    
    ###########################################################
    #### Row 1 Right: Proportion time spent doing various behaviors:
    ''' Consider turning this into its own function.'''
        
    dat = np.mean(np.stack(data['prop_state'],axis=0),axis=0)
    tot = [1,1,1,1]
    labs= [s.capitalize() for s in data['prop_labels'][0]]
    labs = ['Stop', 'Amb','FM','Rear']
    labels=['Pre','Dur','Post']
    width = 0.6      # the width of the bars: can also be len(x) sequence
    #cols=['k','w','b','m']
    cols= [ (0,0,1), (1,1,1), '0.7', (0,0,0)]
    ax=f_row[1][1]
    for i,b in enumerate(labs):
        bt=[0,0,0]
        if i>0:
            bt=np.sum(dat[0:i,:],axis=0)
        ax.bar(labels, dat[i,:], width, label = b,
               edgecolor='k',facecolor=cols[i],bottom=bt)
    ax.legend()
    ax.set_xlim([-1,5])
    ax.set_ylabel('Proportion')
    #2-Way ANOVA:
    # predictors of 'behavior' and 'stim_period'
    
    dat = np.stack(data['prop_state'],axis=0)
    behav=np.array([])
    time=np.array([])
    stim_period=np.array([])
    subject = np.array([])
    for i,b in enumerate(labs):
        pre=dat[:,i,0] # mice x pre
        dur=dat[:,i,1] # mice dur
        post=dat[:,i,2]
        time=np.concatenate((time,pre,dur,post),axis=0)
        
        subject=np.concatenate((subject,np.tile(data['anid'],3)),axis=0)              
        stim_period=np.concatenate((stim_period,
                                    np.repeat(['pre','dur','post'],len(pre))), 
                                    axis=0)
        
        tb=np.repeat(b,len(pre)*3,axis=0)
        behav=np.concatenate((behav,tb),axis=0)
        
    df=pd.DataFrame({'anID' : subject,
                    'behavior': behav,
                     'stim_period': stim_period,
                     'prop':time}) 
    stat_temp={}
    print('\n\n(1,1) Behavior proportions stats:')
    for lab in labs:
        subset=df.iloc[df.behavior.values==lab,:]
        res=pg.rm_anova(dv='prop', 
                        within='stim_period', 
                        subject='anID', 
                        data=subset, detailed=True)
        print('\t%s:' % lab)
        out=tuple(res.loc[0,['DF','F','p-unc']].values)
        print('\tRM-ANOVA: df:%d, f:%3.3f, p:%f' % out)
        anova={}
        anova['anova_rm']=res
        post_hocs=pd.DataFrame({'p-corr': [1,1,1]})
        if res.loc[0,'p-unc'] < 0.05:
            post_hocs=pg.pairwise_ttests(dv='prop',
                                         within='stim_period',
                                         subject='anID',
                                         padjust='fdr_bh',
                                         data=subset)
            print(('\tPOST-HOC:\n' \
                  + '\t\tdur-post p: %1.5f\n' \
                  + '\t\tdur-pre  p: %1.5f\n' \
                  + '\t\tpost-pre p: %1.5f\n') \
                  % tuple(post_hocs.loc[:,'p-corr']))
        else:
            print('\tPost-hocs n.s.\n')
        anova['post_hoc']=post_hocs
        stat_temp[lab]=anova
    
    # model = ols('time ~ C(behavior) + C(stim_period) + C(behavior):C(stim_period)', data=df).fit()
    # anova2=sm.stats.anova_lm(model, typ=2)
    stats[1] += [stat_temp]
    
    # stat_temp={'2WayAnova':anova2}
    #### Row 2 Left: Ambulation bout rate:
    dat= np.stack(data['amb_bout_rate'],axis=0)
    labs= data['prop_labels'][0]
    labels=['Pre','Dur','Post']
    width = 0.4       # the width of the bars: can also be len(x) sequence
    cols=['k','w','g']
    ax=f_row[2][0]
    amb_bouts={'rate':dat}
    mean_bar_plus_conf_clip(amb_bouts,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       )
    ax.set_xlim([-1,3])
    ax.set_ylabel('bouts / s')
    ax.set_title('Amb. bout rate')
    
    #### Row 2 Middle: Amb bout speed:
    dat= np.stack(data['amb_bout_speed'],axis=0)
    labs= data['prop_labels'][0]
    labels=['Pre','Dur','Post']
    width = 0.4       # the width of the bars: can also be len(x) sequence
    cols=['k','w','g']
    ax=f_row[2][1]
    amb_bouts={'rate':dat}
    mean_bar_plus_conf_clip(amb_bouts,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       )
    ax.set_xlim([-1,3])
    ax.set_ylabel('cm / s')
    ax.set_title('Amb. bout speed')
    
    #### Row 2 Right: Amb bout duration
    dat= np.stack(data['amb_bout_dur'],axis=0)
    labs= data['prop_labels'][0]
    labels=['Pre','Dur','Post']
    width = 0.4       # the width of the bars: can also be len(x) sequence
    cols=['k','w','g']
    ax=f_row[2][2]
    amb_bouts={'rate':dat}
    mean_bar_plus_conf_clip(amb_bouts,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       )
    ax.set_xlim([-1,3])
    ax.set_ylabel('seconds')
    ax.set_title('Amb. bout dur.')
    

    # pdb.set_trace()
    
    ## Row 3 Left: Amb CV:
    # dat=np.stack(data['amb_cv'],axis=0)
    # ax=f_row[3][0]
    # bouts={'rate':dat}
    # mean_bar_plus_conf_clip(bouts,['Pre','Dur','Post'],
    #                    use_key='rate',ax=ax,clip_method=False,
    #                    color='k')
    # ax.set_xlim([-1,3])
    # ax.set_ylabel('Amb. CV')
    
    #### Row 3 Left: Immobility bout rate:
    dat= np.stack(data['im_bout_rate'],axis=0)
    labs= data['prop_labels'][0]
    labels=['Pre','Dur','Post']
    width = 0.4       # the width of the bars: can also be len(x) sequence
    cols=['k','w','g']
    ax=f_row[3][0]
    bouts={'rate':dat}
    mean_bar_plus_conf_clip(bouts,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       )
    ax.set_xlim([-1,3])
    ax.set_ylabel('(bouts / s)')
    ax.set_title('Immobility bout rate')
        
    #### Row 3 Middle: Imm. bout dur:
    dat= np.stack(data['im_bout_dur'],axis=0)
    labs= data['prop_labels'][0]
    labels=['Pre','Dur','Post']
    width = 0.4       # the width of the bars: can also be len(x) sequence
    cols=['k','w','g']
    ax=f_row[3][1]
    bouts={'rate':dat}
    mean_bar_plus_conf_clip(bouts,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       )
    ax.set_xlim([-1,3])
    ax.set_ylabel('(s)')
    ax.set_title('Immobility bout dur.')
        
    #### Row 3 Right: Rear bout rate
    if not any(data['has_dlc'] == False):
        dat= np.stack(data['rear_bout_rate'],axis=0)
        labs= data['prop_labels'][0]
        labels=['Pre','Dur','Post']
        width = 0.4       # the width of the bars: can also be len(x) sequence
        cols=['k','w','g']
        ax=f_row[3][2]
        bouts={'rate':dat}
        mean_bar_plus_conf_clip(bouts,['Pre','Dur','Post'],
                            use_key='rate',ax=ax,clip_method=False,
                            )
        ax.set_xlim([-1,3])
        ax.set_ylabel('Rear bouts / s')
    else:
        plt.sca(f_row[2][2])        
        plt.title('Rear scoring incomplete. Check summary["has_dlc"]')    
    
    #### Row 4: Deg bias accumulation
    datx = data.loc[:,'x_trace'].values
    #Note: still neeed to properly identify value based on analysis dur:
    daty = data.loc[:,'turn_trace'].values

    ys = []
    xs = []
    
    if data.side[0]=='Bilateral' and method == 'each_dur':
        t0=np.min(data.stim_dur)
        for x,y in zip(datx,daty):
            ind1 = (x >= -t0) &  (x < t0)
            ys.append(y[ind1])
            xs.append(x[ind1]) 
    else: #Assume they are all the same length (5x30)
        t0= data.stim_dur[0]
        for x,y in zip(datx,daty):
            ind1 = (x >= -t0) &  (x < (t0*2))
            ys.append(y[ind1])
            xs.append(x[ind1]) 


    rot_bias_dur=t0        
    ml= min([len(x) for x in ys])
    i=0
    for x,y in zip(xs,ys):
        if len(y) > ml:
            d=len(y)-ml
            ys[i]=y[d:]
            xs[i]=x[d:]
        i += 1
    # For 10x10 & 10x30, use ys[1]
    # dat=np.stack(data.loc[:,'turn_trace'],axis=0)
    
    dat=np.stack(ys,axis=0)
    # pdb.set_trace()
    u_mice = np.unique(data.anid)
    temp=[]
    for mouse in u_mice:
        ind = np.argwhere(data.loc[:,'anid'].values == mouse)
        d=[]
        for i in ind:
            t=dat[i[0],:]
            if data.loc[i[0],'side']=='Right':
                t=-t
            d.append(t)
        d = np.array(d)
        temp.append(np.mean(d,axis=0))
        y=np.array(temp)
    # pdb.set_trace()
    ax= f_row[4][0]
    mean_cont_plus_conf_array(xs[1],
                                    y.T,
                                    plot_each=True,
                                    highlight=[0,rot_bias_dur,3000],
                                    ax=ax)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(360))

    if data.loc[0,'side'] == 'Bilateral':
        ax.set_ylabel('CW - CCW (deg)')
    else:
        ax.set_ylabel('IPSI - CONTRA (deg)')
    ax.set_xlabel('Time from stim (s)')
   

            
    ### Row 5 Middle: CW or Ipsi rotations:
    
    dat=np.stack(data.loc[:,'ipsi_rot_rate'],axis=0) * 10 #Per 10 seconds
    u_mice=np.unique(data['anid'])
    temp=[]
    #Note: Data must first be averaged per mouse (left & right data)
    for mouse in u_mice:
        ind = data.loc[:,'anid'] == mouse
        temp.append(np.mean(dat[ind,:],axis=0))
    bouts_cw={'rate' : np.array(temp)}
    ax=f_row[5][1]
    mean_bar_plus_conf_clip(bouts_cw,['Pre','Dur','Post'],
                       use_key='rate', ax=ax, clip_method=False,
                       )
    if np.any(data.loc[:,'side']=='Left') \
        or np.any(data.loc[:,'side']=='Right'):
            axis_lab='Ipsi Rot. per 10s'
    else:
        axis_lab = 'CW Rot. per 10s'
        
    ax.set_ylabel(axis_lab)
    ax.set_xlim([-1,3])
    ax.set_ylim([0,3])
    plt.title('n=%d' % (len(temp)))

    #### Row 5 Right: CCW or Contra rotations:
    dat=np.stack(data.loc[:,'contra_rot_rate'],axis=0) * 10 #Per 10 seconds
    temp=[]
    #Note: Data must first be averaged per mouse (left & right data)
    for mouse in u_mice:
        ind = data.loc[:,'anid'] == mouse
        temp.append(np.mean(dat[ind,:],axis=0))
    bouts_ccw={'rate' : np.array(temp)}
    ax=f_row[5][2]
    mean_bar_plus_conf_clip(bouts_ccw,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       )
    if np.any(data.loc[:,'side']=='Left') \
        or np.any(data.loc[:,'side']=='Right'):
            axis_lab='Contra Rot. per 10s'
    else:
        axis_lab = 'CCW Rot. per 10s'
        
    ax.set_ylabel(axis_lab)
    ax.set_xlim([-1,3])
    ax.set_ylim([0,3])
    plt.title('n=%d' % (len(temp)))  
    
    #### Row 5 Left: Ipsi and Contra barplot horizontal:
    use_sns = True
    ax = f_row[5][0]
    if use_sns is False:
        mean_bar_plus_conf_clip(bouts_cw,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       horz = 'Left', color='b',
                       )
        
        mean_bar_plus_conf_clip(bouts_ccw,['Pre','Dur','Post'],
                       use_key='rate',ax=ax,clip_method=False,
                       horz = 'Right',color='r',
                       )
    else:
        #Use seaborn:
        plot_unilateral_rotations_back2back(data,ax=ax)
        ax.yaxis.grid()
        ax.set_xlim([-3,3])
        ax.set_xticks(ticks=[i for i in range(-3,4)])
        ax.set_xticklabels(labels=[str(abs(i)) for i in range(-3,4)])
        ax.set_xlabel('Rotations / 10s')
        ax.set_ylabel('Stimulation period')
        
    # #### Save image option:
    # if save == True:
    #     path_dir = str(meta['pn'][0].parent)
    #     anid= meta['anid'][0]
    #     proto=meta['etho_exp_type'][0]
    #     plt.show(block=False)
    #     plt.savefig(path_dir + '/%s_%s_summary_v%d.png' %  (anid,proto,plt_ver))
    if close == True:
        plt.close()
    else:
        return fig,stats,f_row

def plot_openloop_cond_comparison(data,save=False,close=False):
    '''
    

    Parameters
    ----------
    data : dictionary 
        Key = condition. Value = behavior.open_loop_summary_collect() outputs pd.DataFrames
    save : Boolean, optional
        DESCRIPTION. The default is False.
    close : Boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    fig = plt.figure(constrained_layout = True,figsize=(10,7))
    gs = fig.add_gridspec(2, 5)
    f_row=list(range(gs.get_geometry()[0]))
    conds = [i for i in data.keys()]
    f_row[0]=[fig.add_subplot(gs[0,0:gs.ncols])]
    f_row[1]=[fig.add_subplot(gs[1,i]) for i in range(gs.ncols)]
   
    ax_speedbar=None
    ax=[]
    cols=['b','r']
    for cond in conds:  
        dur = data[cond]['stim_dur'][0] #For now assuming all exp have same duration of stimulation
        dat=data[cond].loc[:,'vel_trace']
        y=np.vstack([d for d in dat])
        ym= np.mean(y,axis=0)        
        x=data[cond].loc[:,'x_trace'].values
        clip_ave={'cont_y' : ym,
                  'cont_x' : x[0],
                  'cont_y_conf' : signals.conf_int_on_matrix(y,axis=0),
                  'disc' : np.vstack(data[cond]['stim_speed'].values)}
                  
        ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                          xlim=[-dur,dur*2],
                                          highlight=[0,dur,25],
                                          ax=f_row[0][0])

    cao=data[cond]['cell_area_opsin'][1]
    lines=ax_speedbar.get_lines()
    for cond,col,line in zip(conds,cols,lines):
        line.set_label(cond)
        line.set_color(col)
    plt.sca(ax_speedbar)
    ax_speedbar.set_ylabel('Speed (cm/s)')
    ax_speedbar.set_xlabel('Time (min)')
    plt.legend()
    
    if 'rear_bout_rate' in data[conds[0]].columns:
        bar_types=['per_mobile','amb_bout_rate','amb_speed','im_bout_rate',
                   'rear_bout_rate']
        ylabs=['Time mobile (%)','Amb bouts / s','Amb speed (cm/s)','Im bouts /s',
               'Rear bouts / s']
    else:        
        bar_types=['per_mobile','amb_bout_rate','amb_speed','im_bout_rate']
        ylabs=['Time mobile (%)','Amb bouts / s','Amb speed (cm/s)','Im bouts /s']
    label_columns=['Pre','Dur','Post'] #Should be determined based on data thirds
    sns.set_theme(style="whitegrid")
    i=0
    # pdb.set_trace()
    for examine,ylab in zip(bar_types,ylabs):
        df = behavior.summary_collect_to_df(data,
                              use_columns=['anid',examine],
                              label_columns=label_columns,
                              var_name='time_window', 
                              value_name=examine,                        
                              static_columns=['anid'])
    
        ax = sns.barplot(ax=f_row[1][i],
                         x="time_window", 
                         y=examine,
                         hue='cond',
                         data=df,)
        if i < 4:
             ax.legend_.remove()
                
       
        ax.set_xlabel('Stim periods (%ds)' % dur)
        ax.set_ylabel(ylab)
        i += 1
        
        
def plot_unilateral_rotations_back2back(data,
                                        use = ['contra_rot_rate','ipsi_rot_rate'],
                                        colors=['r','b'],
                                        ax=None):    
    '''
    Plot horizontal barplots back to back of Ipsi vs. Contra rotations per 10s.

    Parameters
    ----------
    data - pandas.DataFrame() output from function:
        gittislab.experiment_summary_collect()

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
        
    stats: List
        Contains relevant statistics for each panel in the figure by row:
            [[col0, col1, col3], [col0,col12] , [etc]]

    '''
    if ax == None: #If no axis provided, create a new plot
        fig,ax=plt.subplots()
    plt.sca(ax)
    
    mods = [1,-1]
    label_columns = ['Pre','Dur','Post']

    for examine,col,mod in zip(use,colors,mods):    
        dat = table_wrappers.groupby_concat_mean(data,'anid',examine) #Average Left & Right stim responses
        keep= {'data':dat}
        df = behavior.summary_collect_to_df(keep,
                          use_columns=['anid',examine],
                          label_columns=label_columns,
                          var_name='time_window', 
                          value_name=examine,                        
                          static_columns=['anid'],
                          method = 'array')
        df.loc[:,examine] *= mod*10 #Per 10s
        df.loc[:,examine] += mod/1e2 #to offset dots slightly 
        sns.set_color_codes("pastel")
        sns.barplot(x=examine, y='time_window', data=df,
                    label= 'Rotations / 10s', color=col)
        
        sns.set_color_codes("muted")
        
        sns.swarmplot(
                      x=examine,
                      y='time_window',
                      dodge=True,
                      color=col,
                      data=df,
                      size=5,)
        
    if ax == None:
        return fig,ax
    else:
        return ax
    
def plot_zone_day(raw,meta,save=False,close = False):    
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
    plt_ver=3
    fig = plt.figure(constrained_layout = True,figsize=(8.5,11))
    gs = fig.add_gridspec(6, 3)
    f_row=list(range(gs.get_geometry()[0]))
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
    ax_speedbar = mean_cont_plus_conf_clip(clip_ave,
                                      xlim=[-stim_dur,stim_dur*2],
                                      highlight=[0,stim_dur,25],
                                      ax=f_row[2][0])
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Time from stim (s)')
    
    ax_speed = mean_bar_plus_conf_clip(vel_clip,['Pre','Dur','Post'],
                                  ax=f_row[2][1],
                                  )
    
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
    f_row[3][0].fill_between([0,(t[2]-t[1])/60],[2,2],y2=[0,0],
                        color=hl_color, alpha=0.3,edgecolor='none')
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
    f_row[4][0].fill_between([0,(t[2]-t[1])/60],[2,2],y2=[0,0],
                        color= hl_color, alpha=0.3,edgecolor='none')
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
    
    
    #### Row 5: Cross counts & durations vs. time
    width=5
    sz_counts=[]
    sz_durs=[]
    # data=behavior.experiment_summary_helper(raw, meta)
    counts,durs,time= behavior.measure_crossings(raw[sz],fs=meta['fs'][0],
                                        binsize=width, 
                                        ) #analysis_dur=10
    # for i in range(len(t)-1):
    #     ind=(( time >= t[i]) & (time < t[i+1]))
    #     sz_counts.append(counts[ind])
    #     sz_durs.append(durs[ind])
    
    
    x = (time*60 - t[1])/60
    pbins=counts
    baseline= np.nanmean(pbins[x<0])
    pbins=pbins/baseline
    hl_color = 'b'
    f_row[5][0].fill_between([0,(t[2]-t[1])/60],[6,6],y2=[0,0],
                        color= hl_color, alpha=0.3,edgecolor='none')
    f_row[5][0].bar(x,pbins,width-0.5,facecolor='k')
    f_row[5][0].set_ylabel('# SZ Crosses')
    f_row[5][0].set_xlabel('Time (m)')

    f_row[5][0].set_xlim((x[0] - width, x[-1]))
    f_row[5][0].set_ylim((0,4))
    f_row[5][0].plot([x[0]-width,x[-1]],[1,1],'--r')
    
    
    pbins=durs
    baseline= np.nanmean(pbins[x<0])
    pbins=pbins/baseline
    hl_color = 'b'
    f_row[5][1].fill_between([0,(t[2]-t[1])/60],[6,6],y2=[0,0],
                        color= hl_color, alpha=0.3,edgecolor='none')
    f_row[5][1].bar(x,pbins,width-0.5,facecolor='k')
    f_row[5][1].set_ylabel('Crossing durations')
    f_row[5][1].set_xlabel('Time (m)')

    f_row[5][1].set_xlim((x[0] - width, x[-1]))
    f_row[5][1].set_ylim((0,4))
    f_row[5][1].plot([x[0]-width,x[-1]],[1,1],'--r')
    
    #### Save image option:
    if save == True:
        path_dir = str(meta['pn'][0].parent)
        anid= meta['anid'][0]
        proto=meta['etho_exp_type'][0]
        plt.show(block=False)
        plt.savefig(path_dir + '/%s_%s_summary_v%d.png' %  (anid,'zone',plt_ver))
    
    if close == True:
        plt.close()
    else:
        return fig

def plot_zone_mouse_summary(data, 
                            save=False,
                            color='b',
                            close=False,
                            example_mouse=0):    
    '''

    Parameters
    ----------
    data - pandas.DataFrame() output from function:
        gittislab.behavior.experiment_summary_collect()

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
    #### Set up figure axes in desired pattern
    plt_ver = 3

    fig = plt.figure(constrained_layout = True,figsize=(8.5,8.5))
    gs = fig.add_gridspec(4, 3)
    f_row=list(range(gs.get_geometry()[0]))
    
    f_row[0]=[fig.add_subplot(gs[0,i]) for i in range(3)] #example spatial location
    sum_i=1
    
    f_row[1]=[fig.add_subplot(gs[1,i]) for i in range(3)] #Summary of spatial locations
    f_row[2]=[fig.add_subplot(gs[2,i]) for i in range(3)]
    f_row[3]=[fig.add_subplot(gs[3,i]) for i in range(3)]
    
    # f_row[4]=[fig.add_subplot(gs[4,i]) for i in range(3)]
    # f_row[5]=[fig.add_subplot(gs[5,i]) for i in range(3)]
    
    #### Row 0: Example arena exploration of mouse:
    zone = int(data.loc[example_mouse,'proto'].split('_')[1])
    xx=data.loc[example_mouse,'x_task_position']
    yy=data.loc[example_mouse,'y_task_position']
    pre_dur_post_arena_plotter(xx,yy,f_row[0],highlight=zone,color=color)
    cao=data['cell_area_opsin'][example_mouse]
    f_row[0][0].set_title('%s RTPP (%s), n=%d' % (cao,
                                        data['proto'][example_mouse],
                                        data.shape[0]))
        
    anids=np.unique(data['anid'])
    dat=[]
    
    #### Row 1: Mean spatial exploration:
    # First a 1-D  Gaussian
    tt = np.linspace(-35, 35, 30)
    bump = np.exp(-0.1*tt**2)
    bump /= np.trapz(bump) # normalize the integral to 1
    
    # make a 2-D kernel out of it
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    
    anids=np.unique(data['anid'])
    dat=[]
    for anid in anids: #Currently assume each row is a unique animal
        row = data.loc[:,'anid'].values == anid
        zone = int(data.loc[row,'proto'].values[0].split('_')[1])
        temp = data.loc[np.argwhere(row)[0][0],'prob_density_arena']
        keep=list(range(0,3))
        for i,t in enumerate(temp):
            t=np.rot90(t)
            if zone == 2:
                t=np.fliplr(t)
            
            t = scipy.signal.convolve2d(t, kernel, mode='same')
            keep[i]=t
        dat.append(keep)
    dat=np.stack(dat,axis=-1)      
    dat=np.mean(dat,axis=3)

    # pdb.set_trace()
    for i,a in enumerate(f_row[1]):
        d=dat[i,:,:]
        # d=np.rot90(d)
        norm=colors.LogNorm(vmin=np.min(d[:])+0.0001, vmax=np.max(d[:])*0.7)
        a.pcolormesh(d,norm=norm,cmap='coolwarm',shading='auto')
        a.plot([10,10],[0,20],'--w')
        # plt.sca(a)
        # plt.xlabel('cm')
        # plt.ylabel('cm')   
    
    
    #### Row 2: %Time in SZ analysis
    dat=[]
    for anid in anids:
        row = data.loc[:,'anid'].values == anid
        zone = int(data.loc[row,'proto'].values[0].split('_')[1])
        if zone == 1:
            dat.append(data.loc[row,'per_time_z1'].values[0])
        else:
            dat.append(data.loc[row,'per_time_z2'].values[0])
    dat=np.stack(dat) 
    f,h=mean_bar_plus_conf_array(dat, ['Pre','Dur','Post'],ax=f_row[2][0])
    f_row[2][0].set_xlabel('Active RTPP')
    f_row[2][0].set_ylabel('% Time in SZ')   
    
    
    #### Row 3: Left: Stim zone cross counts
    dat_cross=[]
    dat_dur=[]
    norm = True
    width=int(data.loc[0,'zone_1_cross_bin_size'])
    time = np.array([x for x in range(1,30+width,width)]) #Note: using first 10 minutes of pre,dur,post regardless of original duration
    # time = np.array([x for x in range(0,29+width,width)])
    # pdb.set_trace()
    for anid in anids:
        row = data.loc[:,'anid'].values == anid
        zone = int(data.loc[row,'proto'].values[0].split('_')[1])
        if zone == 1:
            c=np.hstack(data.loc[row,'zone_1_cross_counts_binned'].values[0])    
            d=np.hstack(data.loc[row,'zone_1_cross_durs_binned'].values[0])
            # time=np.hstack(data.loc[row,'zone_1_cross_bin_times'].values[0])
        else:
            c=np.hstack(data.loc[row,'zone_2_cross_counts_binned'].values[0])
            d=np.hstack(data.loc[row,'zone_2_cross_durs_binned'].values[0])
            # time=np.hstack(data.loc[row,'zone_2_cross_bin_times'].values[0])
        if norm == True:
            
            if np.any(c[time < 10]):
                c = c / np.nanmean(c[time < 10])
                d = d /  np.nanmean(d[time < 10])
            else:
                c=np.ones(c.shape)
                d=np.ones(d.shape)
            
        dat_dur.append(d)
        dat_cross.append(c)
        
        
    dat_dur=np.vstack(dat_dur)
    dat_cross=np.vstack(dat_cross)
    mc=np.nanmean(dat_cross,axis=0) #Add error bars?
    md=np.nanmean(dat_dur,axis=0) #Add error bars?
        
    t=[0,10,20,30]
    # width=data.loc[0,'zone_1_cross_bin_size']
    
    # time = range(0,30+width,width) #Note: using first 10 minutes of pre,dur,post regardless of original duration
    x = (time - t[1])
    pbins=mc
    pdb.set_trace()
    # baseline= np.nanmean(pbins[x<0])
    # pbins=pbins/baseline
    hl_color = 'b'
    f_row[3][0].fill_between([0,(t[2]-t[1])],[6,6],y2=[0,0],
                        color= hl_color, alpha=0.3,edgecolor='none')
    f_row[3][0].bar(x,pbins,width-0.1,facecolor='k',align='edge')
    f_row[3][0].set_ylabel('# SZ Crosses')
    f_row[3][0].set_xlabel('Time (m)')

    f_row[3][0].set_xlim((x[0] - width, x[-1]))
    f_row[3][0].set_ylim((0,4))
    if norm == True:
        f_row[3][0].plot([x[0]-width,x[-1]],[1,1],'--r')
        f_row[3][1].set_ylim((0,2))
    
    #### Row 3 middle: Stim zone durs
    # pbins=pbins/baseline
    pbins=md
    baseline= np.nanmean(pbins[x<0])
    hl_color = 'b'
    f_row[3][1].fill_between([0,(t[2]-t[1])],[70,70],y2=[0,0],
                        color= hl_color, alpha=0.3,edgecolor='none')
    f_row[3][1].bar(x,pbins,width-0.1,facecolor='k',align='edge')
    f_row[3][1].set_ylabel('SZ Cross Dur (s)')
    f_row[3][1].set_xlabel('Time (m)')

    f_row[3][1].set_xlim((x[0] - width, x[-1]))
    f_row[3][1].set_ylim((0,70))
    if norm == True:
        f_row[3][1].plot([x[0]-width,x[-1]],[1,1],'--r')
        f_row[3][1].set_ylim((0,2))
    
    
    
    
    
    if close == True:
        plt.close()
    else:
        return fig
    
def plot_light_curve_sigmoid(pns,laser_cal_fit,sum_fun, y_col='im',
                             save=False,load_method='raw',fit_method='lm',
                             iter=50):
    '''
    
    Plot 50x2 (multi power short stim experiment) and attempt to fit a sigmoid.

    Parameters
    ----------
    pns : LIST
        Experiment raw.csv Path() variables to use with ethovision_tools.csv_load
        
    laser_cal_fit : numpy.poly1d
        Fit of average relationship between arduino PWM value (x) and raw laser power (y)
    save : BOOLEAN, optional
        Whether to save a copy of this plot in experiment folder. The default is False.
    fit_method : STRING
        Method scipy.optimize.curve_fit uses to fit sigmoid
    Returns
    -------
    keep_x - list of fitted sigmoid x values for each experiment
    keep_y - list of fitted sigmoid y values for each experiment

    '''
    ver = 0
    keep_x=[]
    keep_y=[]
    keep_an=[]
    keep_par = []
    for ii in range(0,len(pns),1):
        raw,meta=ethovision_tools.csv_load(pns[ii],columns='All',method=load_method )
        keep_an.append(meta['anid'][0])
        trial_fn=dataloc.gen_paths_recurse(pns[ii].parent,filetype = 'pwm_output*.csv')
        trials=pd.read_csv(trial_fn)
        doubled=False
        #% Percent Time Immobile
        if len(meta['stim_dur']) > 50:
            doubled=True
        if doubled:
            meta.drop(axis=0,index=range(0,100,2),inplace=True)
            meta.reset_index(inplace=True)
       
        m_clip= behavior.stim_clip_grab(raw,meta,y_col=y_col, 
                                       stim_dur=meta['stim_dur'][0],
                                       baseline = meta['stim_dur'][0],
                                       summarization_fun=sum_fun)
        
        y=m_clip['disc'][:,1]
        x=laser_cal_fit(trials.loc[:,'PWM'].values)
        plt.figure()
        plt.plot(x,y,'ok')
        
        #Bin per mW and average:
        ym=[]
        for i in range(0,8,1):
            ind = (x >= i) & (x < (i+1))
            mbin= np.mean(y[ind])
            ym.append(mbin)
        xm=[i+0.5 for i in range(0,8,1)]
        plt.plot(xm,ym,'or')
        all_base=m_clip['disc'][:,0]
        all_base=all_base[~np.isnan(all_base)]
        # base_zer=np.zeros(all_base.shape)
        base_mean=np.nanmean(all_base)
        plt.plot(0,base_mean,'*b')
        # xf = np.append(x,np.zeros(len(m_clip['disc'][:,0])))
        # yf = np.append(y,m_clip['disc'][:,0])
        
        
        xs=[i/100 for i in range(-100,800,1)]

        all_po=model.bootstrap_model(x,y,
                                    model.fit_sigmoid,
                                    model_method='lm',
                                    iter = iter, 
                                    subsamp_by=5)
        keep_par.append(all_po)
        ys=model.sigmoid(xs, all_po[0], all_po[1], all_po[2],all_po[3])
        plt.plot(xs,ys,'b')
        keep_x.append(xs)
        keep_y.append(ys)
        plt.xlabel('Power (mW)')
        if y_col == 'im':
            plt.ylabel('% Time Immobile')
        
        plt.title('%d, %s %s' % (ii,meta['anid'][0],meta['protocol'][0]))
        plt.xticks(np.arange(0, round(np.max(xs))+1, step=1))
        if save == True:
            path_dir = str(meta['pn'][0].parent)
            anid= meta['anid'][0]
            proto=meta['etho_exp_type'][0]
            plt.show(block=False)
            plt.savefig(path_dir + '/%s_%s_%s_summary_v%d.png' %  (anid,proto,
                                                                   y_col, ver))
    return keep_x, keep_y, keep_an, keep_par

def zone_day_crossing_stats(raw,meta):
    
    ac_on,ac_off= signals.thresh(raw['iz1'].astype(int),0.5,'Pos')
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
    f_row=list(range(gs.get_geometry()[0]))
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

def etho_check_sidecamera(vid_path,frame_array,plot_points=None,
                          part_colors=['.r']):
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
                    for color,part in zip(part_colors,parts):
                        ax[i].plot(part[0],part[1],color,markersize=3)
            else:
                ax[i].set_title('No frame returned.')
        plt.tight_layout()
        plt.show(block=False)
        plt.savefig(frame_dir + '/frameset_%03d.png' % ii )
        plt.close()
    cap.release()
    
def gen_sidecam_plot_points(df,parts,framesets,raw_dlc_file=False):
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
    plot_points=[]
    if raw_dlc_file == True:
        valid_parts=np.unique([col[1] for col in df.columns])
        dims=['x','y']
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
    else:
        valid_parts = np.unique([col for col in df.columns])
        dims=['x','y']
        for frameset in framesets:
            frame_points=[]
            for frame in frameset:
                plot_part=[]
                for p in parts:
                    point=[] #Each ROW of temp will be a point plotted on each frame of frameset
                    for dim in dims:
                        part='%s_%s' % (p,dim)
                        if part not in valid_parts:
                            print('Invalid part %s requested.' % part)
                        else:
                            point.append(df.loc[frame,part])
                    plot_part.append(point)
                frame_points.append(plot_part)
            plot_points.append(frame_points)

    return plot_points
