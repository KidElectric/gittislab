from gittislab import dataloc, mat_file, signals, \
                      ethovision_tools, table_wrappers, model
import numpy as np
import os
import math
from matplotlib import pyplot as plt
import pandas as pd
# import modin.pandas as pd
#import cv2
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy import stats as scistats
from itertools import compress
from pathlib import Path
import pdb 

def smooth_vel(raw,meta,win=10):
    '''
    smooth_vel(raw,meta,win=10)
    Calculate mouse velocity by smoothing (x,y) position coordinates with boxcar convolution
    of a length set by "win".
    Parameters
    ----------
    raw : ethovision dataframe
        raw ethovision experiment dataframe created by ethovision_tools
    meta : dict
        parameters of ethovision experiment created by ethovision_tools
    win : int, optional
        Length of smoothing window in samples. The default is 10. (333ms at 29.97 fps)

    Returns
    -------
    vel : array
        Instantaneous velocity recalculated from smooth (x,y).

    '''
    vel=[]
    fs=meta['fs'][0]
    cutoff=3 #Hz
    x=raw['x']
    x[0:5]=np.nan
    y=raw['y']
    y[0:5]=np.nan
    x_s=signals.pad_lowpass_unpad(x,cutoff,fs,order=5)
    y_s=signals.pad_lowpass_unpad(y,cutoff,fs,order=5)

    #Calculate distance between smoothed (x,y) points for smoother velocity
    for i,x2 in enumerate(x_s):
        x1=x_s[i-1]
        y1=y_s[i-1]
        y2=y_s[i]
        dist=signals.calculateDistance(x1,y1,x2,y2)
        vel.append(dist / (1/fs))
    return np.array(vel)

def preproc_raw(raw,meta,win=10):

    fs=meta['fs'][0]
    cutoff=3 #Hz    
    if 'dlc_top_head_x' in raw.columns:
        has_dlc = True #Always use ethovision by default?    
        im_thresh=0.5 #cm/s more stringent, used in combo with other DLC metrics
    else:
        has_dlc =False
        im_thresh=1 #Velcity thresh in cm/s
    meta['has_dlc'] = has_dlc
    preproc=raw.loc[:,['time','x','y','vel','laserOn']]
    keep_cols=['iz1','iz2','full_rot_cw', 'full_rot_ccw']
    for col in keep_cols:
        if col in raw.columns:
            preproc[col]=raw[col]
    
    x_s=signals.pad_lowpass_unpad(raw['x'],cutoff,fs,order=5)
    y_s=signals.pad_lowpass_unpad(raw['y'],cutoff,fs,order=5)
    # pdb.set_trace()
    
    #Calculate distance between smoothed (x,y) points for smoother velocity
    vel=[]
    dist=[]
    for i,x2 in enumerate(x_s):
        x1=x_s[i-1]
        y1=y_s[i-1]
        y2=y_s[i]
        dist_temp=signals.calculateDistance(x1,y1,x2,y2)
        dist.append(dist_temp)
        vel.append(dist_temp / (1/fs))
    preproc['dist']=dist
    # pdb.set_trace()
    if has_dlc:
        #Add rearing to preproc:
        p_thresh=0.296 #Determined emprically on test set as having FA rate <0.05
        rear_lp = 1 #Hz
        if ('COMPUTERNAME' in os.environ.keys()) and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
            base = Path(r'F:/Users/Gittis/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/')
        else:
            base =Path( '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/')
        rf_model_fn = base.joinpath('rf_model_v3.joblib')
        weights_fn= base.joinpath('bi_rearing_nn_weightsv3')
        tab_fn= base.joinpath('to_nnv3.pkl')
        p_rear, rear = detect_rear_from_model(raw,meta,
                                              prob_thresh=p_thresh,
                                              low_pass_freq=rear_lp,
                                              weights_fn = weights_fn,
                                              tab_fn = tab_fn,
                                              rf_model_fn = rf_model_fn
                                              )
        preproc['prob_rear']=p_rear
        preproc['rear'] = rear
        meta['rear_p_thresh']=p_thresh
        meta['rear_lowpass_freq']=rear_lp        
        meta['bad_dlc_tracking']=False
    else:
        preproc['rear']=np.ones(raw['x'].shape)*np.nan
        
    # if 'vel_smooth_win_ms' not in meta.columns:
    preproc['vel']=vel
    meta['vel_smooth_win_ms']=win/fs * 1000 # ~333ms
                    
    thresh=2; #cm/s; Kravitz & Kreitzer 2010
    # min_bout_dur=0.5; #0.5s used in Kravitz & Kreitzer 2010 #Implement in bout_analyze
    
    raw['rear'] = preproc['rear']
    raw,meta=add_amb_to_raw(raw,meta,
                                amb_thresh = thresh,
                                im_thresh=im_thresh,                
                                use_dlc=has_dlc) #Thresh and dur
    add_cols=['im','im2','amb','amb2','fm', 'fm2']
    
    for col in add_cols:
        if col in raw.columns:
            preproc[col]=raw[col]
    # pdb.set_trace()
    meta['amb_vel_thresh']=thresh
    meta['im_vel_thresh']=im_thresh
    
    #Use smoothed x,y to calculate a normalized position
    is_zone=('zone' in meta['protocol'][0]) 
    is_openloop=('10x' in meta['protocol'][0])
    if  is_zone or (is_openloop and ('iz1' in raw.columns)):
        #Normalize coordinates to zone-cross x value == 0
        cross=np.concatenate(([0],np.diff(raw['iz1'].astype(int)) > 0)).astype('bool')
        if any(cross):
            meta['iz1_center_zero']=True
            zone_cross_x=np.nanmedian(x_s[cross])
            x_s = x_s - zone_cross_x #so that 0 is defined as beginning of zone 1
        else:
            meta['iz1_center_zero']=False
        
    raw['x']=x_s
    raw['y']=y_s
    xn,yn=norm_position(raw)
    preproc['x']=xn
    preproc['y']=yn

    preproc['dir']=smooth_direction(raw,meta,use_dlc=False, win=win)
    if 'dlc_top_head_x' in raw.columns:
        preproc['dlc_dir']=smooth_direction(raw,meta,use_dlc=True,win=win)
    else:
        preproc['dlc_dir']=np.ones(raw['x'].shape)*np.nan
        
    # preproc['meander'] = measure_meander(raw,meta,use_dlc=False)
    return preproc,meta

def add_amb_to_raw(raw,meta,amb_thresh=2, im_thresh=1, use_dlc=False):
    '''
        Add behavioral states: ambulation, immobile, rearing, fine_movement
        im, amb, fm = states calculated from ethovision % pixel change
        im2, amb2, fm2 = states calculated from mouse velocity & deeplabcut positional
                        markers if present
        amb_thresh = cm/s above which mouse is considered ambulating
        im_thresh = cm/s below which mosue is considered immobile (for im2 calc only)
        use_dlc = whether to use deeplabcut approach 
        return raw & meta
    '''
    fs=meta['fs'][0]
    
    amb = raw['vel'] > amb_thresh       
    # amb2=amb
    im = raw['im'].values.astype(float)
    # if use_dlc == True:
    #     if 'rear' in raw.columns:
    #         rear=raw['rear']
    #     elif 'dlc_is_rearing_logical' in raw.columns:
    #         rear = raw['dlc_is_rearing_logical']
            
    #     amb2[rear == True] = False
    #     meta['alt_im_method'] = 'dlc_pos_delta'
    #     im2 = np.zeros(im.shape,dtype=bool)
        
    #     vel = raw['vel'].values
    #     vel[0:5]=np.nan
    #     feats=['dlc_snout_x','dlc_snout_y',
    #            'dlc_side_left_hind_x', 'dlc_side_left_hind_y',
    #            'dlc_side_right_fore_x','dlc_side_right_fore_y', 
    #            'dlc_side_tail_base_x', 'dlc_side_tail_base_y',
    #            'dlc_side_left_fore_x', 'dlc_side_left_fore_y',
    #            'dlc_side_right_hind_x', 'dlc_side_right_hind_y',
    #            'dlc_head_centroid_x', 'dlc_head_centroid_y',]
    #     d=np.ones(im.shape) * 0
    #     step=20
    #     x=raw['dlc_top_body_center_y'].values 
    #     for c in feats:
    #         dat = raw[c].values
    #         dat=signals.max_normalize_per_dist(x,dat,step,poly_order=2) #Correct for distance from camera
    #         dd = abs(np.diff(np.hstack((dat[0],dat))))
    #         d=np.vstack((d,dd))
    #     d=np.nanmax(d,axis=0)

    #     dlc_crit = 0.004 # For position normalized, determined to get good agreement / improvement over ethovision
    #     vel_crit = im_thresh
    #     if 'mouse_height' in raw.columns:
    #         height=raw['mouse_height']
    #     elif 'dlc_front_over_rear_length' in raw.columns:
    #         height = raw['dlc_front_over_rear_length']
    #     # height_crit = 0.3 #Exclude for now
    #     new_crit = np.array((d < dlc_crit) & (vel < vel_crit) ) # & (height < height_crit))
    #     smooth_crit = 0.2
    #     new_crit_temp = signals.boxcar_smooth(new_crit,round(meta['fs'][0]*0.5)) 
    #     im2 =  new_crit_temp >= smooth_crit
    #     im2[rear == True] = False # Say rearing is fine_movement
    # else:
    # im2 = np.zeros(im.shape,dtype=bool)
    # meta['alt_im_method'] = 'vel_thresh'        
    # im2 =raw['vel'] < im_thresh

            

    # raw['im2']=im2 # 'im' is the immobility measure calculated in ethovision itself
    # amb2[im2 == True] = False #If DLC used, rear also made false in amb2
    # raw['amb2']=amb2
    # r,p=scistats.pearsonr(im2,im)
    # meta['im_im2_pearson'] = r
    
    amb[raw['im'] == True] = False
    raw['amb']=amb
    
    raw['fm']= (raw['amb']==False) & (raw['im']==False)
    # raw['fm2'] = (raw['im2']==False) & (raw['amb2'] == False)
    # if use_dlc:
    #     raw['fm2'].values[rear == True ] = True
        
    return raw, meta

def norm_position(raw):
    '''
    Normalize mouse running coordinates so that the center of (x,y) position is
    (0,0). Use method of binning observed (x,y) in cm and finding center bin.
    Note: This could fail if the mouse never moves.
    Returns:
        xn - np.array, normalized x center coordinate of mouse
        yn - np.array, normalized y center coordinate of mouse
    '''
    cx,cy = find_arena_center(raw)        
    x=raw['x'].values
    y=raw['y'].values
    y=y-cy
    x=x-cx
    #Potentially improve this cleaning process with 
    exc= (abs(x) > 25) | (abs(y) > 25)  #Exclude
    x[exc] = np.nan
    y[exc] = np.nan
    xn = x - np.nanmin(x) - (np.nanmax(x) - np.nanmin(x))/2
    yn = y - np.nanmin(y) - (np.nanmax(y) - np.nanmin(y))/2
    return xn,yn

def find_arena_center(raw):
    '''
    Approximate the (x,y) center of the arena based off where the mouse has traveled.
    
    Uses a binning method to estimage endpoints of mouse travel.
    
    Takes ethovision raw dataframe as input.
    
    Output: x_center, y_center -> coordinate of center
    '''
    
    # matlab code:
    # function [cx,cy]=find_arena_center(x,y)

    use = ['x','y']
    for a in use:
        x=raw[a].values
        mm=math.floor(np.nanmin(x))
        mmx=math.ceil(np.nanmax(x))
        bin=[i for i in range(mm,mmx,1)]
        c,_=np.histogram(x,bin)
        c=np.log10(c)
        c[np.isnan(c) | np.isinf(c)]=0
        
        # Find first and last bin > 0 and center on this value:
        ind=[i for i,v in enumerate(c) if v > 0]
        min=np.float(bin[ind[0]])
        mx= np.float(bin[ind[-1]])
        cx=(mx-min)/2 + min
        if a=='x':
            x_center = cx
        elif a == 'y':
            y_center= cx
            
    return x_center,y_center

def z2_to_z1_cross_detect(raw,meta,start_cross_dist=15,stop_cross_dist=10,
                          max_cross_dur=5,min_total_dist=5,min_cross_dist=3):
    z2=raw['iz2'] == True
    facing_z1=((raw['dir'] > 155) | (raw['dir'] < -165))
    close_or_crossing= (raw['x'] < start_cross_dist) & (raw['x'] >-stop_cross_dist)
    facing_close= facing_z1 & close_or_crossing  # & z2 #& raw['ambulation']
    start,stop = signals.thresh(facing_close.astype(int),0.5, sign='Pos')
    cross=[]
    no_cross=[]
    for i,j in zip(start,stop):
        xi=raw['x'][i]
        xj=raw['x'][j]
        fullx=raw['x'][i:j]
        dur = (j-i)/meta['fs'][0]
        dist=signals.calculateDistance(xi,0,xj,0)
        good_dur=((dur < max_cross_dur) and (dur > 0.5))
        if (dist > min_total_dist) and (xi > 0) and good_dur:
            if (xj < 0) or any(fullx < -min_cross_dist):
                cross.append([i,j])
            else:
                no_cross.append([i,j])
    return cross, no_cross

def trial_part_count_cross(cross,non_cross,meta):
    ''' Break trial into parts (pre, dur, post) and count crossings in each period
        (Make customized time binning? )
        
        Uses cross and non_cross output from z2_to_z1_cross_detect()
    '''
    t=[i for i in range(4)]
    t[0]=0
    t[1]=meta.task_start[0] * meta['fs'][0]
    t[2]=meta.task_stop[0]* meta['fs'][0]
    t[3]=meta.exp_end[0]* meta['fs'][0]
    tot_c=np.zeros((2,3))

    for i in range(len(t)-1):
        in_period=0
        for c in cross:
            if (c[0] >=t[i]) and (c[1] < t[i+1]):
                in_period += 1
        tot_c[0,i]=in_period
        in_period=0
        for c in non_cross:
            if (c[0] >=t[i]) and (c[1] < t[i+1]):
                in_period += 1
        tot_c[1,i]=in_period
    return tot_c

def trial_part_position(raw,meta,chunk_method='task'):
    # x,y=norm_position(raw) #Already performed with preprocessed raw dataframe
    x=raw['x']
    y=raw['y']
    t=[i for i in range(4)]
    
    t[0]=0
    if chunk_method == 'task':        
        t[1]=meta.task_start[0]
        t[2]=meta.task_stop[0]       
        t[3]=meta.exp_end[0]
        wins = [[t[0], t[1]],
                [t[1], t[2]],
                [t[2], t[3]]]
    elif chunk_method == 'thirds':
        third=meta.exp_end[0]/3
        t[1]=1*third
        t[2]=2*third
        t[3]=meta.exp_end[0]
        wins = [[t[0], t[1]],
                [t[1], t[2]],
                [t[2], t[3]]]
    elif chunk_method == '10min' :
        t[1] = meta.task_start[0]
        t[2] = meta.task_stop[0]
        
        wins=[[0, 10*60],
              [t[1], t[1]+(10*60)],
              [t[2], t[2]+(10*60)], 
              ]
    xx=[]
    yy=[]
    for win in wins:
        ind=(raw['time']>= win[0]) & (raw['time'] < win[1])
        xx.append(x[ind])
        yy.append(y[ind])
    return xx,yy

def stim_xy_loc(raw,meta):
    '''
    Return normalized (x,y) coordinates of stimulation onsets
    '''
    x_on=[]
    y_on=[]
    xx,yy = norm_position(raw)
    x=xx[1:]
    y=yy[1:]
    for i,on in enumerate(meta['stim_on']):
        ind=raw['time'] > on
        d = np.diff(ind) == 1
        x_on.append(x[d])
        y_on.append(y[d])
    meta['stim_on_x']=x_on
    meta['stim_on_y']=y_on
    return meta

def measure_bearing(raw,meta):
    # Measurement revolves around direction of travel when crossing midline of open field
    # and subsequent continuation along same route or rebounding
    
    return raw['time'].values[0]

def measure_crossings(in_zone_data,fs, binsize=1, analysis_dur=10):
    '''
    Measure number and duration of crosses into specified zone of 
    open field during 'zone' (RTPP) task.

    Parameters
    ----------
    in_zone_data: np.array of int
        Logical array turned to int of whether mouse is in zone of interest
    binsize: int
        Bin size to use (minutes)
    analysis_dur: int
        Duration of time to use during task and post period (assumes pre period is 10minutes)

    Returns
    -------
    binned_counts : array
        Binned cross counts into use_zone.
    med_dur : array
        Binned median cross durations (in seconds)
    keep_t : array
        Center of bins used.

    '''
    
    ac_on,ac_off= signals.thresh(in_zone_data,0.5,'Pos')
    min=0 #min cross duration
    all_cross=[]
    cross_t=[]
    for on,off in zip(ac_on,ac_off):
        if (off-on) > min:
            all_cross.append([on,off])
            cross_t.append(on/fs)
    durs = np.diff(np.array(all_cross),axis=1) / fs
    # print('%d crossings detected. Median dur: %1.2fs' % \
    #       (len(all_cross),np.median(durs)))
    
        
    
    maxt = round((len(in_zone_data)/fs)/60)
    t0=0
    all_on=(np.array(all_cross)[:,0]/fs) / 60 #in minutes
    time_bins = np.array([x for x in range (t0,maxt+binsize,binsize)])
    binned_counts=[]
    med_dur=[]
    keep_t=[]
    
    for t0,t1 in zip(time_bins[0:-1],time_bins[1:]):
        ind = (all_on > t0) & (all_on <=t1)
        binned_counts.append(np.sum(ind))
        if any(ind):
            med_dur.append(np.median(durs[ind]))
        else:
            med_dur.append(np.nan)
        
        keep_t.append(t0+((t1-t0)/2))
    

    return np.array(binned_counts),np.array(med_dur),np.array(keep_t)
    
def measure_meander(raw,meta,use_dlc=False):
    '''
    Change in direction vs. change in distance traveled.
    
    '''
        
    fs=meta['fs'][0]
    # dir = smooth_direction(raw,meta,use_dlc=use_dlc)
    dir=raw['dir']
    diff_angle=signals.angle_vector_delta(dir[0:-1],dir[1:],thresh=20,fs=fs)
    dist = raw['vel'] * (1 / meta.fs[0]) #Smoothed version of distance
    dist[0:3]=np.nan
    meander = diff_angle / (dist[1:])
    return meander

def smooth_direction(raw,meta,
                     use_dlc=False,
                     win=10):
    '''
    

    Parameters
    ----------
    raw : pd.DataFrame() 
        Contains columns of ethovision / DLC raw tracking data via ethovision_tools.unify_to_csv()
    meta : pd.DataFrame()
        Contains columns of experiment parameters data ethovision_tools.unify_to_csv()
    head_tail : List of columns to use for calculating direction (default is ethovision)
        DESCRIPTION. The default is ['x_nose','y_nose','x_tail','y_tail'].
    win : Int, optional, currently unused
        DESCRIPTION. The default is 10.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if use_dlc == True:
        head_tail=['dlc_top_head_x','dlc_top_head_y',
           'dlc_top_tail_base_x','dlc_top_tail_base_y']
        multiplier= -1
    else:
        head_tail=['x_nose','y_nose','x_tail','y_tail']
        multiplier=1
        
    #Interpolate NaNs with nearyby values and convert to floats:
    cutoff= 3 #Hz
    fs=meta['fs'][0]
    x_n=signals.pad_lowpass_unpad(raw[head_tail[0]],cutoff,fs,order=5)
    y_n=signals.pad_lowpass_unpad(raw[head_tail[1]],cutoff,fs,order=5)
    x_t=signals.pad_lowpass_unpad(raw[head_tail[2]],cutoff,fs,order=5)
    y_t=signals.pad_lowpass_unpad(raw[head_tail[3]],cutoff,fs,order=5)
    
    #Calculate distance between smoothed (x,y) points for smoother body angle
    angle=[]
    for x1,y1,x2,y2 in zip(x_n,y_n,x_t,y_t):
        angle.append(signals.one_line_angle(x2,y2,x1,y1))
    
    return np.array(angle)*multiplier
def measure_rotations(raw,meta,
                      max_dur=10, 
                      min_dur = 0.5,
                      min_rot = 90,
                      rot_thresh = 340):
    
    '''
        RETURN: 
            cw = -1 when mouse turned CCW within min_dur to max_dur time
               =  1 when mouse turned CW within time window
    '''
    fs=meta['fs'][0]
    d = np.rad2deg(np.unwrap(np.deg2rad(raw.loc[:,'dir'])))
    d = d - d[0]
    ad = np.abs(d)
    a=0
    cw = np.zeros(d.shape)
    while np.any(ad > rot_thresh):
        cross = np.argwhere(ad > rot_thresh)
        if np.any(cross):
            b= int(cross[0])
            dur = (b - a) / fs
            if (dur < max_dur) and (dur > min_dur):
                if d[b] < 0:
                    cw[(a+1):(b-1)] = 1
                else:
                    cw[(a+1):(b-1)] = -1
            d= d- d[b]
            d[0:b] = 0
            # pdb.set_trace()
            a = b
            ad = np.abs(d)
    return cw
   
def measure_directedness(raw,meta):
    '''
    Change in direction vs. change in distance traveled.
    
    '''
    fs=meta['fs'][0]
    dir = raw['dir']
    diff_angle=signals.angle_vector_delta(dir[0:-1],dir[1:],thresh=20,fs=fs)

    dist = raw['vel'] * (1 / meta.fs[0]) #Smoothed version of distance
    dist[0:3]=np.nan
    directed = (100*dist[1:]) / diff_angle # meter travel / change in direction
    return directed

def stim_clip_grab(raw,meta,y_col,x_col='time',
                   stim_dur=30, baseline = None,
                   summarization_fun=np.nanmean):
    
    if isinstance(meta['fs'],float):
        fs=meta['fs']
    else:
        fs=meta['fs'][0]
    if baseline == None: # Default: create symmetric around stimulus 30-30-30 : Pre-Dur-Post
        baseline = stim_dur
    nsamps=math.ceil(((baseline*2) + stim_dur) * fs)
    ntrials=len(meta['stim_on'])
    cont_y_array=np.empty((nsamps,ntrials))
    cont_y_array[:]=np.nan
    y=raw[y_col].values
    x=raw[x_col].values
    disc_y_array=np.empty((ntrials,3)) #Pre Dur Post
    for i,on_time in enumerate(meta['stim_on']):
        # Array containing continuous part of analysis:
        off_time=meta['stim_off'][i]
        base_samp= int(round((on_time - baseline) * fs))
        on_samp = int(round(on_time * fs))
        on_time_samp= int(round((on_time + stim_dur) * fs))
        off_samp = int(round(off_time * fs))
        post_samp = int(round((off_time + baseline) * fs))
        intervals=[[base_samp,on_samp],
                   [on_samp,on_time_samp],
                   [off_samp,post_samp]]
        if base_samp >= 0:
            cont_y_array[:,i]=y[base_samp:(base_samp + nsamps)]

        if i==0:
            cont_x_array=x[base_samp:(base_samp + nsamps)] - on_time
        # (1,3) Array containing Pre, Dur, Post discretized analysis:
        for ii,interval in enumerate(intervals):
            disc_y_array[i][ii]=summarization_fun(y[interval[0]:interval[1]])

    out_struct={'cont_x':cont_x_array,'cont_y':cont_y_array,'disc':disc_y_array,
                'samp_int':intervals}
    return out_struct


def stim_clip_average(clip,continuous_y_key='cont_y',discrete_key='disc'):
    '''
    stim_clip_average(out_struct) Returns an average +/- 95% conf of continuous
           and discrete fields of this structure

    Parameters
    ----------
    out_struct : Struct.
        Output structure from stim_clip_grab with a continous field ('cont') and discrete field ('disc')

    Returns
    -------
    out_struct_averaged : TYPE
        DESCRIPTION.

    '''
    confidence = 0.95
    continous_x_key=continuous_y_key.split('_')[0] + '_x'
    out_ave={'disc_m':np.empty((3,1)),
             'disc_conf':np.empty((3,1)),
             'cont_y':np.empty(clip[continuous_y_key][:,0].shape),
             'cont_x':np.empty(clip[continous_x_key].shape),
             'cont_y_conf':np.empty(clip[continuous_y_key][:,0].shape)}
    n=clip[continuous_y_key].shape[1]
    
    for i,data in enumerate(clip[discrete_key].T):
        m = np.nanmean(data)
        std_err = np.nanstd(data)/np.sqrt(n)
        h = std_err * scistats.t.ppf((1 + confidence) / 2, n - 1)
        out_ave['disc_m'][i]=m
        out_ave['disc_conf'][i]=h
    # out_ave['disc']=clip[discrete_key]
    y=clip[continuous_y_key]
    ym=np.mean(y,axis=1)
    out_ave['cont_y']=ym
   
    # std_err = np.nanstd(clip[continuous_y_key],axis=1)/np.sqrt(n)
    # h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    out_ave['cont_y_conf']=signals.conf_int_on_matrix(y)
    out_ave['cont_x']=clip[continous_x_key]
    
    return out_ave

def open_loop_summary_collect(basepath,
                              conds_inc=[],
                              conds_exc=[],
                              update_rear=False,
                              stim_analyze_dur=10):
    '''
    

    Parameters
    ----------
    basepath : TYPE
        DESCRIPTION.
    conds_inc : TYPE, optional
        DESCRIPTION. The default is [].
    conds_exc : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    data : pandas.DataFrame()
        Each row is an experiment day
        Columns include all info relevant for plotting a summary day using:
            gittislab.plots.plot_openloop_mouse_summary(data)
        DESCRIPTION.

    '''
    min_bout= 0.5
    
    data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'stim_dur','vel_trace','amb_speed','amb_bouts','per_mobile'])
    use_cols=['time','vel','im','dir','ambulation','meander']
    
    #### Loop through experimental mice to load & process all data
    version=3 #Version where versioning is formally implemented (all at version 3)
    # version = 4 #Change rear threshold to 0.6 and min thresh to 0.55
    
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        csv_paths=dataloc.raw_csv(basepath,inc,exc)

        if isinstance(csv_paths,Path):
            csv_paths=[csv_paths]
        if len(csv_paths) == 0:
            print('No files found.')
            # pdb.set_trace()
        for ii,path in enumerate(csv_paths):            
            print('Inc[%d], file %d) %s loaded...' % (i,ii,str(path)))
            raw,meta=ethovision_tools.csv_load(path,method='preproc')

            if update_rear == True:
                raw = update_rear_logic(raw)
            temp = experiment_summary_helper(raw,meta,
                                             min_bout=min_bout,
                                             update_rear=update_rear,
                                             stim_analyze_dur=stim_analyze_dur)
            data=data.append(temp,ignore_index=True)
            
    # data.sort_values('anid',inplace=True,ignore_index=True)
    return data

def update_rear_logic(raw):
    old_fields=['im','fm','amb']
    for f in old_fields:
        ind = raw.loc[:,'rear'].values
        ind[np.isnan(ind)]=False
        raw.loc[ind,f]=False
    return raw

def zone_rtpp_summary_collect(basepath, conds_inc=[],conds_exc=[],
                              min_bout=0.5,
                              bin_size=10,
                              update_rear=False,
                              stim_analyze_dur=10,
                              zone_analyze_dur= 10 * 60):
    '''
    
    '''
    data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'stim_dur','vel_trace','amb_speed','amb_bouts',
                              'per_mobile','per_time_z1', 'per_time_z2',
                              'prob_density_edges', 'prob_density_arena'])
    use_cols=['time','vel','im','ambulation','meander','iz1','iz2']
    
    #### Loop through experimental mice to load & process all data
    version=3 #Version where versioning is formally implemented (all at version 3)
    
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        csv_paths=dataloc.raw_csv(basepath,inc,exc)
        if isinstance(csv_paths,Path):
            csv_paths=[csv_paths]
        for ii,path in enumerate(csv_paths):            
            print('Inc[%d], file %d) %s loaded...' % (i,ii,str(path)))
            raw,meta=ethovision_tools.csv_load(path,method='preproc')

            temp = experiment_summary_helper(raw,meta,
                                             min_bout=min_bout,
                                             bin_size = bin_size,
                                             stim_analyze_dur=stim_analyze_dur,
                                             zone_analyze_dur = zone_analyze_dur)
            
            data=data.append(temp,ignore_index=True)
    return data

def free_running_summary_collect(basepath,conds_inc=[],conds_exc=[],
                                 min_bout= 0.5,bin_size=10):
    '''
    Take in list of experiment tags to include & exclude, analyze them
    if they are free-running days

    Parameters
    ----------
    basepath : TYPE
        DESCRIPTION.
    conds_inc : TYPE, optional
        DESCRIPTION. The default is [].
    conds_exc : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    data : pandas.DataFrame()
        Each row is an experiment day
        Columns include all info relevant for plotting a summary day using:
            gittislab.plots.plot_openloop_mouse_summary(data)
        DESCRIPTION.

    '''
    
    
    
    data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                              'stim_dur','vel_trace','amb_speed','amb_bouts',
                              'per_mobile','per_time_z1','per_time_z2'])
    use_cols=['time','vel','im','dir','ambulation','meander']
    
    #### Loop through experimental mice to load & process all data
    version=3 #Version where versioning is formally implemented (all at version 3)
    
    # version = 4 #Change rear threshold to 0.6 and min thresh to 0.55
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        csv_paths=dataloc.raw_csv(basepath,inc,exc)
        if isinstance(csv_paths,Path):
            csv_paths=[csv_paths]
        for ii,path in enumerate(csv_paths):            
            print('Inc[%d], file %d) %s loaded...' % (i,ii,str(path)))
            raw,meta=ethovision_tools.csv_load(path,method='preproc')
            if meta['no_trial_structure'][0] == True:
                #Analyze in thirds by adding a "stim" at 1/3 through:
                pseudo_stim = meta
                third=meta['exp_end'][0]/3
                pseudo_stim['stim_on'] = third
                pseudo_stim['stim_off'] = 2*third
                pseudo_stim['stim_dur']=third
                
                temp = experiment_summary_helper(raw,pseudo_stim,
                                                 min_bout=min_bout,
                                                 bin_size = bin_size)
                data=data.append(temp,ignore_index=True)
    return data

def summary_collect_to_df(summary_dict,
                          use_columns,
                          label_columns,
                          var_name, 
                          value_name,                        
                          static_columns,
                          sort_column=None,
                          method='list'):
    df_conds=summary_dict.keys()
    dfs= []
    for cond_label in df_conds:
        temp=summary_dict[cond_label].loc[:,use_columns]
        if not(sort_column == None):
            temp=temp.sort_values(by=[sort_column])
        if method == 'list':
            temp2=pd.DataFrame(temp[value_name].to_list(),
                               columns=label_columns)
        else:
            temp2=pd.DataFrame(np.vstack(temp[value_name]),
                               columns=label_columns)
            
        for col in static_columns:
            temp2[col]=temp[col]
        dfs.append(temp2)
    # pdb.set_trace()
    out = table_wrappers.df_melt_stack(dfs,df_conds,
                                       label_columns=label_columns,
                                       var_name=var_name,
                                       value_name=value_name,                        
                                       static_columns=static_columns,
                                       sort_column=sort_column)
    return out

def experiment_summary_helper(raw,
                              meta,
                              min_bout=0.5,
                              bin_size = 10,
                              update_rear=False,
                              stim_analyze_dur=10,
                              zone_analyze_dur= 10 *60):
    temp={}
    temp['anid']=meta['anid'][0]
    temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                          meta['stim_area'][0],
                                          meta['opsin_type'][0])
    temp['proto']=meta['protocol'][0]
    stim_dur = round(np.nanmean(meta['stim_dur']))
    baseline = stim_dur
    if stim_analyze_dur == 'mean':
        stim_analyze_dur = stim_dur
    temp['stim_dur'] = stim_dur
    temp['file_path']=meta['pn'][0]
    free_running = meta['no_trial_structure'][0]
    percentage = lambda x: (np.nansum(x)/len(x))*100
    temp['analysis_stim_dur'] = stim_analyze_dur
    temp['has_dlc']=meta['has_dlc'][0]
    fs=meta['fs'][0]
    #### Calculate stim-triggered speed changes:
    if not free_running:        
        vel_clip=stim_clip_grab(raw,meta,
                                y_col='vel',
                                stim_dur = stim_analyze_dur,
                                baseline = stim_analyze_dur)
        
        clip_ave=stim_clip_average(vel_clip)
        temp['stim_speed']=clip_ave['disc_m'].T
    
        vel_clip=stim_clip_grab(raw,meta,
                                y_col='vel',
                                stim_dur = stim_dur,
                                baseline = baseline)
        clip_ave=stim_clip_average(vel_clip)
        temp['vel_trace']=np.median(vel_clip['cont_y'],axis=1)
        temp['x_trace']=clip_ave['cont_x']
        

        

    else:
        # bin_size = 10 #seconds
        #Instead of calculating metrics relative to stimulation,
        #include entire clips
        temp['vel_trace'] = raw['vel'] #Entire clip
        temp['x_trace']=raw['time'] 
        x,y = signals.bin_analyze(raw['time'],raw['vel'],
                                  bin_dur=bin_size, #Seconds
                                  fun = np.nanmedian)
        temp['vel_bin']=y
        temp['bin_size']=bin_size
        mobile = ~raw['im']
        x,y = signals.bin_analyze(raw['time'],mobile,
                          bin_dur=bin_size,
                          fun = percentage)
        temp['x_bin']=x
        temp['raw_per_mobile']=y
        
 
    
    #### Calculate stim-triggered %time mobile:        
    raw['m']=~raw['im']
    m_clip=stim_clip_grab(raw,meta,y_col='m', 
                                   stim_dur= stim_analyze_dur,
                                   baseline = stim_analyze_dur,
                                   summarization_fun=percentage)
    temp['per_mobile']=np.nanmean(m_clip['disc'],axis=0)
    
    
    # Examine ambulation bouts:
    amb_bouts=bout_analyze(raw,meta,'amb',
                        stim_dur = stim_analyze_dur,
                        min_bout_dur_s=min_bout)
    # pdb.set_trace()
    temp['amb_bout_speed']=np.nanmean(amb_bouts['speed'],axis=0)
    temp['amb_bout_rate']=np.nanmean(amb_bouts['rate'],axis=0)
    temp['amb_bout_speed_cv']=np.nanmean(amb_bouts['cv'],axis=0)
    temp['amb_bout_dur'] = np.nanmean(amb_bouts['dur'],axis=0)
    
    
    im_bouts=bout_analyze(raw,meta,'im',
                        stim_dur=stim_analyze_dur,
                        min_bout_dur_s=min_bout)
    temp['im_bout_rate']=np.nanmean(im_bouts['rate'],axis=0)
    temp['im_bout_dur']=np.nanmean(im_bouts['dur'],axis=0)
    
    
    #Examine rear rate /dur if evailable:
    if 'rear' in raw.columns:
        if any(~np.isnan(raw['rear'])) and np.any(raw['rear']):
            rear_bouts=bout_analyze(raw,meta,'rear',
                        stim_dur = stim_analyze_dur,
                        min_bout_dur_s=min_bout)
    
            temp['rear_bout_rate']=np.nanmean(rear_bouts['rate'],axis=0)
        else:
            temp['rear_bout_rate']=np.zeros((1,3))
    
    #Examine stim-triggered ipsi and contra rotations:
    cw = measure_rotations(raw,meta) #currently may drop first rotation if occurs at the end of 10s window
    if meta.loc[0,'side'] == 'Left':
        ipsi = cw == -1  #CCW
        contra = cw == 1 #CW
    elif meta.loc[0,'side'] == 'Right':
        ipsi = cw == 1 #CW
        contra = cw == -1 #CCW
    else: #Bilateral trial, but can still look at cw & ccw rotations:
        ipsi = cw == 1 #CW rotations stored ipsi
        contra = cw == -1 # CCW rotations stored in contra
    ipsi[np.isnan(ipsi)] = 0
    contra[np.isnan(contra)]=0
    raw['ipsi'] = ipsi.astype(bool)
    raw['contra'] = contra.astype(bool)
    
    #Isolate rotations > 0.3s & < 10s:
    temp['side'] = meta.loc[0,'side']
    min_rot_dur = 0.3 # seconds
    max_rot_dur = 10 # seconds
    if np.any(ipsi):
        ipsi_rot_bouts=bout_analyze(raw,meta,'ipsi',
                                    stim_dur = stim_analyze_dur,
                                    min_bout_spacing_s = 0,
                                    min_bout_dur_s = min_rot_dur,
                                    max_bout_dur_s = max_rot_dur)
        
        temp['ipsi_rot_rate']=np.nanmean(ipsi_rot_bouts['rate'],axis=0)
    else:
        temp['ipsi_rot_rate']=np.array([0,0,0])
    
    if np.any(contra):
        contra_rot_bouts=bout_analyze(raw,meta,'contra',
                                    stim_dur = stim_analyze_dur,
                                    min_bout_spacing_s = 0,
                                    min_bout_dur_s = min_rot_dur,
                                    max_bout_dur_s = max_rot_dur)
        temp['contra_rot_rate']=np.nanmean(contra_rot_bouts['rate'],axis=0)
    else:
        temp['contra_rot_rate']=np.array([0,0,0])
        
    # pdb.set_trace()
    ### Calculate stim-triggered Proportion: FM, AMB, IM
    #pdb.set_trace()
    use = ['im','amb','fm']
    if update_rear == True:
        use += ['rear']
    collect=[]
    for col in use:       
        clip=stim_clip_grab(raw,meta,y_col=col, 
                                   stim_dur = stim_analyze_dur,
                                   baseline = stim_analyze_dur,
                                   summarization_fun=np.nansum)
        collect.append(np.nansum(clip['disc'],axis=0))
    collect=np.vstack(collect)
    tot=np.sum(collect,axis=0)
    
    temp['prop_state']=collect / tot
    temp['prop_labels'] = use
    
    #Perform statistics on these differences
    
    ### Calculate % time in each zone:
    t=[i for i in range(3)]
    if zone_analyze_dur == 'mean': #Currently not implemented        
        t[0]=[0,meta.task_start[0]]
        t[1]=[meta.task_start[0],meta.task_stop[0]]
        t[2]=[meta.task_stop[0],meta.exp_end[0]]
    else:
        t[0]=[0, zone_analyze_dur]
        t[1]=[meta.task_start[0], meta.task_start[0] + zone_analyze_dur ]
        t[2]=[meta.task_stop[0], meta.task_stop[0] + zone_analyze_dur]
    
    in_zone1=[]
    in_zone2=[]
    if 'iz1' in raw:
        z1=np.array(raw.loc[:,'iz1'].astype(int))
        z2=np.array(raw.loc[:,'iz2'].astype(int))
    else:
        z1=np.array(raw.loc[:,'x']<0).astype(int)
        z2=np.array(raw.loc[:,'x']>0).astype(int)        
    for ts in t:
        ind=(raw.loc[:,'time']>= ts[0]) & (raw.loc[:,'time'] < ts[1])
        in_zone1.append(percentage(z1[ind]))
        in_zone2.append(percentage(z2[ind]))
        
    temp['per_time_z1']=in_zone1
    temp['per_time_z2']=in_zone2
    
    #Measure crossing:
    cross_bin=1 #minute
    z1_counts,z1_durs,z1_time= measure_crossings(z1,fs,
                                        binsize=cross_bin, 
                                        analysis_dur=zone_analyze_dur)
    temp['zone_1_cross_counts_binned']=z1_counts
    temp['zone_1_cross_durs_binned']=z1_durs
    temp['zone_1_cross_bin_times']=z1_time
    temp['zone_1_cross_bin_size']=cross_bin
    
    z2_counts,z2_durs,z2_time= measure_crossings(z2,fs,
                                        binsize=cross_bin, 
                                        analysis_dur=zone_analyze_dur)
    temp['zone_2_cross_counts_binned']=z2_counts
    temp['zone_2_cross_durs_binned']=z2_durs
    temp['zone_2_cross_bin_times']=z2_time
    temp['zone_2_cross_bin_size']=cross_bin
    
    #Chunk mouse locations:
    temp['zone_analyze_dur'] = zone_analyze_dur
    if zone_analyze_dur == 'mean':
        xx,yy=trial_part_position(raw,meta, chunk_method='task')
    else:
        xx,yy=trial_part_position(raw,meta, chunk_method='10min') 
        
    
    temp['x_task_position']=xx #Pre, during, post
    temp['y_task_position']=yy #Pre, during, post
    
    #Convert 2 2d histogram:
    
    hist=[]
    for x,y in zip(xx,yy):
        dat,xbin,ybin=np.histogram2d(x, y, bins=20, range=[[-25,25],[-25,25]], density=True)
        # dat = np.log10(dat)
        # dat[np.isinf(dat)]=0
        hist.append(dat) #Log10(Probability Density)
    temp['prob_density_edges']=xbin
    temp['prob_density_arena']=hist
    return temp

def experiment_summary_saver(data,path=[],use_cols=[],
                             groupby=False,method='compact'):
    meta_cols=['anid','proto','cell_area_opsin','side','file_path',
               'stim_dur', 'analysis_stim_dur','has_dlc','zone_analyze_dur']
    pdp_labs = ['prestim','durstim','poststim']
    pdp_cols = [ 'stim_speed', 'amb_speed', 'amb_bouts', 'per_mobile',
                'amb_bout_rate', 'amb_cv','contra_rot_rate',
                'im_bout_rate', 'ipsi_rot_rate', 'per_time_z1', 
                'per_time_z2','rear_bout_rate','prop_state',]
    
    per_ten_s = ['ipsi_rot_rate','contra_rot_rate']
    new_dat = pd.DataFrame(data.loc[:,meta_cols])
    temp = new_dat['analysis_stim_dur'].values[:,np.newaxis]
    if 'zone_' not in data.loc[:,'proto'][0]:
        for lab in pdp_labs:
            new_dat.loc[:,'%s_analysis_dur' % lab] = temp
    
        new_dat.drop(columns=['analysis_stim_dur'],inplace=True)
    
    uni = np.any(data.loc[:,'side']=='Left') or np.any(data.loc[:,'side']=='Right')
    if method == 'compact':
        if uni == True:
            use_cols=['ipsi_rot_rate','contra_rot_rate']
            groupby= True #In this case, average ipsi/contra in 'Left' and 'Right' experiments.
    elif len(use_cols) == 0:
        use_cols=pdp_cols
        groupby = False
    # pdb.set_trace()
    
    if 'prop_state' in use_cols: # Separate out into multiple columns:
        prop_dict = {}
        for i,lab in enumerate(data.prop_labels[0]):
            newlab = 'prop_'+ lab
            for ii,win in enumerate(pdp_labs):
                temp=np.stack(data.loc[:,'prop_state'],axis=2)[i,:,:].T
                save_col=newlab
                new_dat.loc[:,'%s_%s' % (win,save_col)] = temp[:,ii]
        use_cols.remove('prop_state')
    for col in use_cols:
        if col in pdp_cols:
            temp = np.vstack(data.loc[:,col])
            
            for i,win in enumerate(pdp_labs):
                if col in per_ten_s:
                    save_col = col + '_per_10s'
                    mod = 10
                else:
                    save_col = col
                    mod = 1
                    
                new_dat.loc[:,'%s_%s' % (win,save_col)] = temp[:,i]*mod
        else:
            temp = data.loc[:,col]
            new_dat.loc[:,col]=temp
    if groupby == True:
       new_dat = new_dat.groupby(by='anid').mean().reset_index() 
    if len(path) > 0:
        print('Save not yet implemented.')
    new_dat =new_dat.sort_values('anid').reset_index()
    return new_dat

def bout_analyze(raw,meta,y_col,stim_dur=10,
                 min_bout_dur_s=0.5,
                 min_bout_spacing_s=0.1,
                 max_bout_dur_s = 1e6,
                 use_dlc=False,
                 calc_meander = False):

    y_col_bout=y_col + '_bout'
    dat=raw[y_col].astype(int) #Note: discretely smoothed by signals.join_gaps
    
    bout_onset = np.concatenate((np.array([0]),np.diff(dat)>0))
    onset_samps = list(compress(range(len(bout_onset)),bout_onset))
    bout_offset = np.concatenate((np.array([0]),np.diff(dat)<0))
    offset_samps= list(compress(range(len(bout_offset)),bout_offset))
    
    #Edgecase: bout starts before recording, so offsets precede onsets:
    if onset_samps[0] > offset_samps[0]:
        #Simplest solution, remove first offset, aligning with remaining onsets
        #(Does not rule out other problems that may arise!)
        offset_samps.pop(0)
    offset_samps=np.array(offset_samps)
    onset_samps=np.array(onset_samps)
    
    #Filter bouts for obeying a minimum bout spacing apart (by default 0.25s):
    min_bout_spacing_samps=round(meta.fs[0]*min_bout_spacing_s)
    new_on,new_off=signals.join_gaps(onset_samps,offset_samps,min_bout_spacing_samps)
    
    
    
    #Also remove detected bouts from bout_onset arrays:
    # bout_onset[[x for x in onset_samps if x not in new_on]]=0 #quite slow
    # bout_offset[[x for x in offset_samps if x not in new_off]]=0
    ind=signals.ismember(new_on, onset_samps) #could this improve speed?
    bout_onset[onset_samps[~ind]]=0
    ind=signals.ismember(new_off,offset_samps) #could this improve speed?
    bout_offset[offset_samps[~ind]]=0
    
    
    onset_samps=new_on
    offset_samps=new_off

    #Filter bouts for obeying a minimum bout duration (by default, 1s):
    dur=np.zeros(dat.shape)
    keep=np.zeros(dat.shape)
    for on,off in zip(onset_samps,offset_samps): #zip() aligns to shortest of onset/offset
        dur_temp=(off-on)/meta.fs[0]
        if (dur_temp >= min_bout_dur_s) and (dur_temp < max_bout_dur_s):
            dur[on]=dur_temp #Bout duration in seconds
        else:
            #do not include this bout
            bout_onset[on]=0 
            bout_offset[off]=0
    raw[y_col_bout] = bout_onset 
    # if y_col == 'ipsi':
    #     pdb.set_trace()
    #Use "_bout" column to calculate discrete # of occurences (min duration?)
    bout_disc=stim_clip_grab(raw,meta,y_col_bout,stim_dur=stim_dur,
                   summarization_fun=np.nansum)
    
    #Use boolean clips as continuous measure of behavior occurence (1= occurring)
    bout_continuous=stim_clip_grab(raw,meta,y_col,stim_dur=stim_dur,
                   summarization_fun=np.nansum)
    
    #Combine these so clip dictionary contains correct discrete and continuous assessment
    #of bout occurence:
    clip=bout_disc
    clip['count'] = clip.pop('disc')
    clip['analyzed']=clip.pop('cont_y') #Indicate onset position of bouts used for analysis
    clip['cont_y']=bout_continuous['cont_y'] #needs to be updated to reflect smoothing above
    
    
    #Add in a 'dur_y' field
    dur[dur==0] = np.nan
    raw['bout_dur']=dur
    bout_dur=stim_clip_grab(raw,meta,'bout_dur',stim_dur=stim_dur,
                   summarization_fun=np.nanmedian)
    clip['dur']=bout_dur['disc']
    
    #Add in a rate of events:
    clip['rate']=clip['count']/stim_dur
    
    #Add in a measure of meandering:
    if calc_meander == True:
        meander=np.empty(dur.shape)
        meander[:]=np.nan
        directedness=meander
        full_meander = raw['meander']
        for on,off in zip(onset_samps,offset_samps):
            meander[on:off]=full_meander[on:off]
            directedness[on:off]=1/full_meander[on:off]
        raw['bout_meander']=meander
        raw['bout_directed']=directedness
        bout_meander=stim_clip_grab(raw,meta,'bout_meander',stim_dur=stim_dur,
                                    summarization_fun=np.nanmedian)
        bout_directed=stim_clip_grab(raw,meta,'bout_directed',stim_dur=stim_dur,
                                    summarization_fun=np.nanmedian)
        clip['meander']=bout_meander['disc']
        clip['directed']=bout_directed['disc']
    
    #Add in a measure of speed:
    speed=np.empty(dur.shape)
    speed[:]=np.nan
    for on,off in zip(onset_samps,offset_samps):
        speed[on:off]=raw['vel'][on:off]
    raw['bout_speed']=speed
    bout_speed=stim_clip_grab(raw,meta,'bout_speed',stim_dur=stim_dur,
                                summarization_fun=np.nanmedian)
    clip['speed']=bout_speed['disc']
    
    cv = lambda x: np.nanstd(signals.log_modulus(x))/np.nanmean(signals.log_modulus(x))
    
    #Add in measure of CV:
    bout_speed=stim_clip_grab(raw,meta,'bout_speed',stim_dur=stim_dur,
                                summarization_fun=cv)
    clip['cv']=bout_speed['disc']
    
    return clip

def mouse_stim_vel(raw,meta,stim_dur=10):
    '''
        mouse_stim_vel(raw,meta)
            Create average velocity trace from one mouse across stimulations.
    Parameters
    ----------
    raw : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    raw_col='vel'
    out_struct=stim_clip_grab(raw,meta,raw_col,stim_dur=stim_dur,summarization_fun=np.nanmedian)
    return out_struct

def load_and_clean_dlc_h5(dlc_h5_path, dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1):
    df = pd.read_hdf(dlc_h5_path)
    
    # Clean up rows based on likelihood:
    like_thresh=dlc_likelihood_thresh
    bad_detect=[]
    for col in df.columns:
        if col[2] == 'likelihood':
            ex = df[col] < like_thresh
            xcol=(col[0],col[1],'x')
            ycol=(col[0],col[1],'y')

            df[xcol][ex]=np.nan
            df[ycol][ex]=np.nan
            if sum(ex) > (len(df[xcol])/2):
                bad_detect.append(xcol)
                print('\t Bad %s %s column' % xcol[1:])
    
    if len(bad_detect) < 1:    
        # Cleanup rows based on outliers:
        sd_thresh= dlc_outlier_thresh_sd #sd
        for col in df.columns:
            if col[2]=='x' or col[2]=='y':
                m=np.nanmean(df[col])
                sd=np.nanstd(df[col])
                ex=(df[col] > (m + sd_thresh * sd))
                df[col][ex]=np.nan
                ex=(df[col] < (m - sd_thresh * sd))
                df[col][ex]=np.nan
        
        print('Data loaded and cleaned')
    
        # Calculate head, front and rear centroids (mean coord):
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
        # x1=df[(exp,'head_centroid','x')].values
        y1=df[(exp,'head_centroid','y')].values
        # x2=df[(exp,'rear_centroid','x')].values
        y2=df[(exp,'rear_centroid','y')].values
        
        # #Correct for distance from camera:
        # out=[]
        step=20
        x=df[(exp,'top_body_center','y')].values
        
        # Front over rear and correct for distance from camera:
        newy=signals.max_normalize_per_dist(x,y2-y1,step,poly_order=2)
        ind=np.array([i for i in range(0,len(newy))])
        
        # Spline-fitting smooth method (slow!):
        print('Quadratic interpolation...')
        ex=np.isnan(newy)
        s=interp1d(ind[~ex],newy[~ex],kind='quadratic',bounds_error=False) 
        smooth_y=s(ind)
        print('\tfinished')
        col=(exp,'front_over_rear','length')
        df[col]=smooth_y
        print('Mouse height added to dataframe')
    else:
        df=[]
    return df

def detect_rear_from_model(raw,meta,prob_thresh=0.5,low_pass_freq=None, 
                       weights_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/bi_rearing_nn_weightsv2',
                       tab_fn='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/to_nnv2.pkl',
                       rf_model_fn = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/DLC Examples/train_rear_model/to_nnv3.pkl'
                       ):
    p_rear,rear_logical = model.nn_rf_predict_from_raw(raw,meta,
                                 prob_thresh=prob_thresh,
                                 low_pass_freq=low_pass_freq, 
                                 weights_fn=weights_fn,
                                 tab_fn=tab_fn,
                                 rf_model_fn = rf_model_fn
                                 )
    start,stop = signals.thresh(rear_logical,0.5,'Pos')
    print("\u001b[36m Detected %d rears in this video \u001b[0m" % len(start))
   
    return p_rear, rear_logical


def detect_rear_from_mouseheight(df,rear_thresh=0.65,min_thresh=0.25,save_figs=False):    
    '''
    

    Parameters
    ----------
    dlc_h5_path : String
        DESCRIPTION. Path to the .h5 file containing deeplabcut video analysis 
        with the following body parts tracked from Ethovision side camera:
            'snout','side_head','side_left_fore','side_right_fore',
            'side_tail_base','side_left_hind','side_right_hind'
    rear_thresh : Integer, optional
        DESCRIPTION. Threshold distance of mouse front over mouse hind 
            used to detect rearing events in mice via gittislab.signals.peak_start_stop() 
            The default is 0.65.
    min_thresh : Integer, optional
        DESCRIPTION. Threshold distance of mouse front over mouse hind
            used to detect when rearing starts/stops via gittislab.signals.peak_start_stop() 
            The default is 0.25.
    save_figs : Boolean, optional
        DESCRIPTION. Save a figure of rear detection. The default is False.
    dlc_outlier_thresh_sd : Inteer, optional
        DESCRIPTION. Threshold (in standard deviations) used to detect when instantaneous changes in 
            deeplabcut position tracking outputs are too large and likely due 
            to mistracking. The default is 4.
    dlc_likelihood_thresh : Int. value between 0 and 1, optional
        DESCRIPTION. Threshold for deeplabcut position probability estimate to be included
            for further analysis. The default is 0.1.

    Returns
    -------
    peaks : array
        index of peak rear times in video samples
    start_peak : array
        index of rear start times in video samples
    stop_peak : array
        index of rear stop times in video samples
    df : dataframe
        full dataframe of deeplabcut analysis with rearing columns added:
            head_centroid = mean of snout and side-head points
            front_centroid = mean of  'snout','side_head','side_left_fore','side_right_fore'
            rear_centroid = mean of 'side_tail_base','side_left_hind','side_right_hind'
            front_over_rear = distance between front and rear centroids, smoothed and used for rearing calc
            is_rearing = logical index of when mouse is rearing given rear_thresh and min_thresh input criteria

    '''
  
    # df = load_and_clean_dlc_h5(dlc_h5_path, dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1)
    
    # Peak - detection method:
    exp=df.columns[0][0]
    dims=['x','y']
    col=(exp,'front_over_rear','length')
    mouse_height=df[col]
    
    # start_peak,stop_peak = signals.thresh(mouse_height,rear_thresh,'Pos')
    peaks,start_peak,stop_peak = signals.expand_peak_start_stop(mouse_height,height=rear_thresh,min_thresh=min_thresh)
    rear=np.zeros(mouse_height.shape)
    for start,stop in zip(start_peak,stop_peak):
        rear[start:stop]=1
    df[(exp,'is_rearing','logical')]=rear
    print('Sanity check')
    print("\u001b[36m Detected %d rears in this video \u001b[0m" % len(start_peak))
   
    # if save_figs==True:
    #     vid_path=dataloc.video(dlc_h5_path.parent)
    #     print(str(vid_path))
    #     rear_dir=str(dlc_h5_path.parent) + '/Rears'
    #     if os.path.exists(rear_dir)==False:
    #         os.mkdir(rear_dir)
    #         print('Made Path')
    #     cap = cv2.VideoCapture(str(vid_path))
    #     if (cap.isOpened()== False): 
    #         print("Error opening video stream or file... skipping attempt")
    #         return peaks,start_peak,stop_peak,df
    #     else:
    #         width  = cap.get(3) # Pixel width of video
    #         height = cap.get(4) # Pixel height of video
    #         fs = cap.get(5) # Sampling rate of video
    #     for pp,peak in enumerate(peaks):
    #         fig,ax=plt.subplots(1,4,figsize=(15,5))    
    #         frames=[start_peak[pp], peak, stop_peak[pp]]
    #         if any(np.array(frames)<0):
    #             print('Negative frame requesteds')
    #         parts=['head_centroid','rear_centroid']
    #         dims=['x','y']
    #         cols=['y.','b.']
    #         for i,f in enumerate(frames):
    #             cap.set(1,f)
    #             ret, frame = cap.read()
    #             if ret == True:
    #                 ax[i].imshow(frame)
    #                 ax[i].set_title('Frame %d, %2.1fs in' % (f,(f-peak)/fs))
    #                 ax[i].set_xlim(width/2,width)
    #                 if height <= 480:
    #                     ax[i].set_ylim(0,height/2)
    #                     ax[i].invert_yaxis()
    #                 for pn,part in enumerate(parts):
    #                     temp=np.zeros((2,1))
    #                     for ii,dim in enumerate(dims):
    #                         temp[ii]=df[(exp,part,dim)][f]
    #                     ax[i].plot(temp[0],temp[1],cols[pn],markersize=3)
    #             else:
    #                 ax[i].set_title('No frame returned.')
    #         ax[-1].plot(mouse_height[frames[0]:frames[2]],'r')
    #         mid=frames[1]-frames[0]
    #         ax[-1].plot(mid,mouse_height[frames[1]],'bo')
    #         ax[-1].plot(0,mouse_height[frames[0]],'rx')
    #         ax[-1].plot(frames[2]-frames[0],mouse_height[frames[2]],'gx')
    #         plt.tight_layout()
    #         plt.show(block=False)
    #         plt.savefig(rear_dir + '/rear_%03d.png' % pp )
    #         plt.close()
    #     cap.release()
    return df

def boris_to_logical_vector(raw,boris,obs_name,evt_start,evt_stop):
    out=np.zeros(raw['time'].values.shape).astype(bool)
    evts=boris['observations'][obs_name]['events']
    dur=[]
    for i in range(0,len(evts)):
        if evts[i][2] == evt_start:
            k = i
            while evt_stop not in evts[k][2]:
                k +=1 
            start=evts[i][0]
            stop=evts[k][0]
            dur.append(stop-start)
            ind= (raw['time'] >=start ) & (raw['time'] < stop)
            out[ind]=True
    return out

def prob_rear_stim_dict(basepath,conds_inc,conds_exc,labels,use_move=True):
    ''' 
    prob_rear_dict() uses behavior.detect_rear() to calculate the probability 
                     of rearing during light stimulation compared to no light stimulation
    Inputs: 
        basepath = str path to top-level data directory, e.g. '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior'
        conds_inc = list of lists of string identifiers used to identify each desired condition for analysis
                e.g. [['GPe', 'Arch', 'CAG', '10x30', 'Bilateral', 'AG'],  # <- one experimental condition with several mice
                     ['Str', '10x30', 'A2A', 'ChR2', 'Bilateral', 'AG']] #<- a second experimental condition with several mice
                see: gittislab.dataloc.common_paths() for more detailed examples
        conds_exc = same as conds_inc but for strings that shape included experiments via exclusion (see gittislab.dataloc for more info)
        labels = list of strings matching conds_inc that describe condition succinctly:
                e.g. labels = ['GPe-CAG-Arch',
                               'Str-A2a-ChR2'] # for conds_inc described above
    # NOTE: an example set of inputs can be generated with:
        basepath,conds_inc,conds_exc,labels=gittislab.dataloc.common_paths() 

    Output:
        
        out = a dictionary of 2-column arrays where key is 'labels' input
            and each entry contains a nx2 array of rearing probabilities.
            
            if use_move = False:
            rows = individual mice
            columns =  [0] is P(Rear) with no stimulation
                       [1] is P(Rear) with light stimulation 
            if use_move = True:
                same as above except also require immobility ==0 (not immobile)
    '''

    use_move=True
    
    #Generate dictionary to store results:
    # if 'out' not in locals():
    out=dict.fromkeys(labels,[])
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        h5_paths=dataloc.gen_paths_recurse(basepath,inc,exc,'.h5')
        out[labels[i]]=np.zeros((len(h5_paths),2))
        for ii,path in enumerate(h5_paths):
            matpath=dataloc.rawmat(path.parent)
            if matpath:
                print('%s:\n\t.h5: %s\n\t.mat: %s' % (labels[i],path,matpath))
                peak,start,stop,df = detect_rear_from_mouseheight(path,rear_thresh=0.7,min_thresh=0.2,save_figs=False,
                        dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1)
                mat=mat_file.load(matpath)
                laserOn=mat['laserOn'][:,0]
                is_rear=df[df.columns[-1]].values
                
                #Take the shorter of the two:
                min_len=min([len(laserOn),len(is_rear)])
                
                if use_move == True:
                    isMove=mat['im'][0:min_len,0]==0 #'im' = immobile. This selects all periods where mouse is NOT immobile.
                    
                    #Calculate probability of rearing when laser is off and mouse is moving:
                    denom_ind =( laserOn[0:min_len]==0) & isMove
                    rear_ind = is_rear[0:min_len]==1
                    out[labels[i]][ii][0]=sum(denom_ind & rear_ind) / sum(denom_ind)
                       
                    #Calculate probability of rearing when laser is on and mouse is moving:
                    denom_ind = (laserOn[0:min_len]==1) & isMove
                    out[labels[i]][ii][1]=sum(denom_ind & rear_ind) / sum(denom_ind)
                else:
                    #Calculate probability of rearing when laser is off:
                    out[labels[i]][ii][0]=sum((laserOn[0:min_len]==0) & (is_rear[0:min_len]==1))\
                        / sum(laserOn[0:min_len]==0)
                    #Calculate probability of rearing when laser is on:
                    out[labels[i]][ii][1]=sum((laserOn[0:min_len]==1) & (is_rear[0:min_len]==1))\
                        / sum(laserOn[0:min_len]==1)
            else:
                print('\n\n NO .MAT FILE FOUND IN %s! \n\n' % path.parent)
            
            print('\n\n')
    return out

def rear_rate_stim_dict(basepath,conds_inc,conds_exc,labels,use_move=True):
    ''' 
    rear_rate_stim_dict() uses behavior.detect_rear() to calculate the rate
                     of rearing during light stimulation compared to no light stimulation
    Inputs: 
        basepath = str path to top-level data directory, e.g. '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior'
        conds_inc = list of lists of string identifiers used to identify each desired condition for analysis
                e.g. [['GPe', 'Arch', 'CAG', '10x30', 'Bilateral', 'AG'],  # <- one experimental condition with several mice
                     ['Str', '10x30', 'A2A', 'ChR2', 'Bilateral', 'AG']] #<- a second experimental condition with several mice
                see: gittislab.dataloc.common_paths() for more detailed examples
        conds_exc = same as conds_inc but for strings that shape included experiments via exclusion (see gittislab.dataloc for more info)
        labels = list of strings matching conds_inc that describe condition succinctly:
                e.g. labels = ['GPe-CAG-Arch',
                               'Str-A2a-ChR2'] # for conds_inc described above
    # NOTE: an example set of inputs can be generated with:
        basepath,conds_inc,conds_exc,labels=gittislab.dataloc.common_paths() 

    Output:
        
        out = a dictionary of 2-column arrays where key is 'labels' input
            and each entry contains a nx2 array of rearing rates.
            
            if use_move = False:
            rows = individual mice
            columns =  [0] is P(Rear) with no stimulation
                       [1] is P(Rear) with light stimulation 
            if use_move = True:
                same as above except also require immobility ==0 (not immobile)
    '''

    use_move=True
    
    #Generate dictionary to store results:
    out=dict.fromkeys(labels,[])
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        h5_paths=dataloc.gen_paths_recurse(basepath,inc,exc,'.h5')
        out[labels[i]]=np.zeros((len(h5_paths),2))
        for ii,path in enumerate(h5_paths):
            matpath=dataloc.rawmat(path.parent)
            if matpath:
                print('%s:\n\t.h5: %s\n\t.mat: %s' % (labels[i],path,matpath))
                peak_array,start,stop,df = detect_rear_from_mouseheight(path,rear_thresh=0.7,min_thresh=0.2,save_figs=False,
                        dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1)
                mat=mat_file.load(matpath)
                laserOn=mat['laserOn'][:,0]
                is_rear=df[df.columns[-1]].values
                # has_fs=[key for key in mat.keys() if 'fs' in key]
                if 'fs' not in mat:# len(has_fs)==0:
                    fs=29.97
                else:
                    fs=mat['fs'][0][0]
                
                #Take the shorter of the two:
                min_len=min([len(laserOn),len(is_rear)])
                peak=[]
                for p in peak_array:
                    if p < min_len:
                        peak.append(p)
                if use_move == True:
                    isMove=mat['im'][0:min_len,0]==0 #'im' = immobile. This selects all periods where mouse is NOT immobile.
                    
                    #Calculate rate of rearing when laser is off and mouse is moving:
                    denom_ind =(laserOn[0:min_len]==0) & isMove
                    nostim_rear=denom_ind[peak]
                    out[labels[i]][ii][0]=sum(nostim_rear) / (sum(denom_ind)/fs)
                       
                    #Calculate rate of rearing when laser is on and mouse is moving:
                    denom_ind = (laserOn[0:min_len]==1) & isMove
                    stim_rear=denom_ind[peak]
                    out[labels[i]][ii][1]=sum(stim_rear) / (sum(denom_ind)/fs)
                else:
                    #Calculate rate of rearing when laser is off:
                    denom_ind=laserOn[0:min_len]==0
                    out[labels[i]][ii][0]= sum(denom_ind[peak]) / (sum(denom_ind)/fs)
                    
                    #Calculate rate of rearing when laser is on:
                    denom_ind=laserOn[0:min_len]==1
                    out[labels[i]][ii][1]= sum(denom_ind[peak]) / (sum(denom_ind)/fs)
            else:
                print('\n\n NO .MAT FILE FOUND IN %s! \n\n' % path.parent)
            
            print('\n\n')
    return out
def prob_rear(is_rear,laserOn,window):
    print('Empty')