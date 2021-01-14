from gittislab import dataloc, mat_file, signal, ethovision_tools
import numpy as np
import os
import math
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.stats import t
from itertools import compress

def smooth_vel(raw,params,win=10):
    '''
    smooth_vel(raw,params,win=10)
    Calculate mouse velocity by smoothing (x,y) position coordinates with boxcar convolution
    of a length set by "win".
    Parameters
    ----------
    raw : ethovision dataframe
        raw ethovision experiment dataframe created by ethovision_tools
    params : dict
        parameters of ethovision experiment created by ethovision_tools
    win : int, optional
        Length of smoothing window in samples. The default is 10. (333ms at 29.97 fps)

    Returns
    -------
    vel : array
        Instantaneous velocity recalculated from smooth (x,y).

    '''
    vel=[]
    fs=params['fs']
    cutoff=3 #Hz
    
    #Interpolate NaNs with nearyby values and convert to floats:
    x=raw['x'].interpolate(method='pad').astype(np.float)
    y=raw['y'].interpolate(method='pad').astype(np.float)
    
    #Low-pass filter method:     
    pad=round(fs*2)
    x=np.pad(x,pad_width=(pad,),mode='linear_ramp')
    y=np.pad(y,pad_width=(pad,),mode='linear_ramp')
    
    x_s=signal.butter_lowpass_filtfilt(x, cutoff, fs, order=5)
    x_s=x_s[pad:-pad]
    y_s=signal.butter_lowpass_filtfilt(y, cutoff, fs, order=5)
    y_s=y_s[pad:-pad]

    #Boxcar method: (doesn't remove stimulation artifacts very well)
    # x_s=signal.boxcar_smooth(raw['x'].values,win)
    # y_s=signal.boxcar_smooth(raw['y'].values,win)
    for i,x2 in enumerate(x_s):
        x1=x_s[i-1]
        y1=y_s[i-1]
        y2=y_s[i]
        dist=signal.calculateDistance(x1,y1,x2,y2)
        vel.append(dist / (1/fs))
    return np.array(vel)

def norm_position(raw_df):
    '''
    Normalize mouse running coordinates so that the center of (x,y) position is
    (0,0). Use method of binning observed (x,y) in cm and finding center bin.
    
    Returns:
        xn - np.array, normalized x center coordinate of mouse
        yn - np.array, normalized y center coordinate of mouse
    '''
    cx,cy = find_arena_center(raw_df)        
    x=raw_df['x'].values
    y=raw_df['y'].values
    y=y-cy
    x=x-cx
    #Potentially improve this cleaning process with 
    exc= (abs(x) > 25) | (abs(y) > 25)  #Exclude
    x[exc] = np.nan
    y[exc] = np.nan
    xn = x - np.nanmin(x) - (np.nanmax(x) - np.nanmin(x))/2
    yn = y - np.nanmin(y) - (np.nanmax(y) - np.nanmin(y))/2
    return xn,yn

def find_arena_center(raw_df):
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
        x=raw_df[a].values
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

def trial_part_position(raw_df,raw_par):
    x,y=norm_position(raw_df)
    t=[i for i in range(4)]
    t[0]=0
    t[1]=raw_par.task_start[0]
    t[2]=raw_par.task_stop[0]
    t[3]=raw_par.exp_end[0]
    xx=[]
    yy=[]
    for i in range(len(t)-1):
        ind=(raw_df['time']>= t[i]) & (raw_df['time'] < t[i+1])
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

def measure_bearing(raw_df,raw_par):
    # Measurement revolves around direction of travel when crossing midline of open field
    # and subsequent continuation along same route or rebounding
    return raw_df['time'].values[0]


def stim_clip_grab(raw_df,raw_par,y_col,x_col='time',stim_dur=30,summarization_fun=np.nanmean):
    
    if isinstance(raw_par['fs'],float):
        fs=raw_par['fs']
    else:
        fs=raw_par['fs'][0]
    baseline=stim_dur
    nsamps=math.ceil(((baseline*2) + stim_dur) * fs)
    ntrials=len(raw_par['stim_on'])
    cont_y_array=np.empty((nsamps,ntrials))
    cont_y_array[:]=np.nan
    y=raw_df[y_col].values
    x=raw_df[x_col].values
    disc_y_array=np.empty((ntrials,3)) #Pre Dur Post
    for i,on_time in enumerate(raw_par['stim_on']):
        # Array containing continuous part of analysis:
        off_time=raw_par['stim_off'][i]
        base_samp= round((on_time - baseline) * fs)
        on_samp = round(on_time * fs)
        on_time_samp= round((on_time + stim_dur) * fs)
        off_samp = round(off_time * fs)
        post_samp = round((off_time + baseline) * fs)
        intervals=[[base_samp,on_samp],
                   [on_samp,on_time_samp],
                   [off_samp,post_samp]]
        
        cont_y_array[:,i]=y[base_samp:(base_samp + nsamps)]
        if i==0:
            cont_x_array=x[base_samp:(base_samp + nsamps)] - on_time
        # (1,3) Array containing Pre, Dur, Post discretized analysis:
        for ii,interval in enumerate(intervals):
            disc_y_array[i][ii]=summarization_fun(y[interval[0]:interval[1]])

    out_struct={'cont_x':cont_x_array,'cont_y':cont_y_array,'disc':disc_y_array,
                'samp_int':intervals}
    return out_struct

def bout_counter(raw,meta,y_col,stim_dur=30,min_bout_dur_s=1,min_bout_spacing_s=0.5):

    y_col_bout=y_col + '_bout'
    dat=raw[y_col].astype(int) #Note: discretely smoothed by signal.join_gaps
    
    bout_onset = np.concatenate((np.array([0]),np.diff(dat)>0))
    onset_samps = list(compress(range(len(bout_onset)),bout_onset))
    bout_offset = np.concatenate((np.array([0]),np.diff(dat)<0))
    offset_samps= list(compress(range(len(bout_offset)),bout_offset))
    
    #Edgecase: bout starts before recording, so offsets precede onsets:
    if onset_samps[0] > offset_samps[0]:
        #Simplest solution, remove first offset, aligning with remaining onsets
        #(Does not rule out other problems that may arise!)
        offset_samps.pop(0)
        
    #Filter bouts for obeying a minimum bout spacing apart (by default 0.5s):
    min_bout_spacing_samps=round(meta.fs[0]*min_bout_spacing_s)
    new_on,new_off=signal.join_gaps(onset_samps,offset_samps,min_bout_spacing_samps)
    #Also remove detected bouts from bout_onset arrays:
    bout_onset[[x for x in onset_samps if x not in new_on]]=0
    bout_offset[[x for x in offset_samps if x not in new_off]]=0
    onset_samps=new_on
    offset_samps=new_off

    #Filter bouts for obeying a minimum bout duration (by default, 1s):
    dur=np.zeros(dat.shape)
    keep=np.zeros(dat.shape)
    for on,off in zip(onset_samps,offset_samps): #zip() aligns to shortest of onset/offset
        dur_temp=(off-on)/meta.fs[0]
        if dur_temp >= min_bout_dur_s:
            dur[on]=dur_temp #Bout duration in seconds
        else:
            #do not include this bout
            bout_onset[on]=0 
            bout_offset[off]=0
    raw[y_col_bout] = bout_onset 
    
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
    
    return clip



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
        m = np.mean(data)
        std_err = np.nanstd(data)/np.sqrt(n)
        h = std_err * t.ppf((1 + confidence) / 2, n - 1)
        out_ave['disc_m'][i]=m
        out_ave['disc_conf'][i]=h
    
    ym=np.mean(clip[continuous_y_key],axis=1)
    out_ave['cont_y']=ym
   
    std_err = np.nanstd(clip[continuous_y_key],axis=1)/np.sqrt(n)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    out_ave['cont_y_conf']=np.array([ym-h, ym+h]).T
    out_ave['cont_x']=clip[continous_x_key]
    
    return out_ave
    
def mouse_stim_vel(raw_df,raw_par,stim_dur=10):
    '''
        mouse_stim_vel(raw_df,params)
            Create average velocity trace from one mouse across stimulations.
    Parameters
    ----------
    raw_df : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    raw_col='vel'
    out_struct=stim_clip_grab(raw_df,raw_par,raw_col,stim_dur=stim_dur,summarization_fun=np.nanmedian)
    return out_struct

def detect_rear(dlc_h5_path,rear_thresh=0.65,min_thresh=0.25,save_figs=False,
                dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1):    
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
            used to detect rearing events in mice via gittislab.signal.peak_start_stop() 
            The default is 0.65.
    min_thresh : Integer, optional
        DESCRIPTION. Threshold distance of mouse front over mouse hind
            used to detect when rearing starts/stops via gittislab.signal.peak_start_stop() 
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
    # from scipy.ndimage import gaussian_filter
    
    df = pd.read_hdf(dlc_h5_path)
    
    # Clean up rows based on likelihood:
    like_thresh=dlc_likelihood_thresh
    for col in df.columns:
        if col[2] == 'likelihood':
            ex = df[col] < like_thresh
            xcol=(col[0],col[1],'x')
            ycol=(col[0],col[1],'y')
            df[xcol][ex]=np.nan
            df[ycol][ex]=np.nan
    
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
    
    # mouse_length=[]
    # for i,r in enumerate(x1):
    #     mouse_length.append(signal.calculateDistance(x1[i],y1[i],x2[i],y2[i]))
        
    # #Correct for distance from camera:
    # out=[]
    step=20
    x=df[(exp,'top_body_center','y')].values
    # col=(exp,'body','length')
    # df[col]=signal.max_correct(x,mouse_length,step,poly_order=2)
    
    # Front over rear and correct for distance from camera:
    newy=signal.max_correct(x,y2-y1,step,poly_order=2)
    ind=np.array([i for i in range(0,len(newy))])
    
    # Spline-fitting smooth method (slow!):
    print('Quadratic interpolation...')
    ex=np.isnan(newy)
    s=interp1d(ind[~ex],newy[~ex],kind='quadratic') #Takes a hecking long time
    smooth_y=s(ind)
    print('\tfinished')
    col=(exp,'front_over_rear','length')
    df[col]=smooth_y
    print('Mouse height added to dataframe')
    
    # Peak - detection method:
    peaks,start_peak,stop_peak = signal.peak_start_stop(smooth_y,height=rear_thresh,min_thresh=min_thresh)
    rear=np.zeros(smooth_y.shape)
    for i,start in enumerate(start_peak):
        rear[start:stop_peak[i]]=1
    df[(exp,'is_rearing','logical')]=rear
    print("\u001b[36m Detected %d rears in this video \u001b[0m" % len(peaks))
   
    if save_figs==True:
        vid_path=dataloc.video(dlc_h5_path.parent)
        print(str(vid_path))
        rear_dir=str(dlc_h5_path.parent) + '/Rears'
        if os.path.exists(rear_dir)==False:
            os.mkdir(rear_dir)
            print('Made Path')
        cap = cv2.VideoCapture(str(vid_path))
        if (cap.isOpened()== False): 
            print("Error opening video stream or file... skipping attempt")
            return peaks,start_peak,stop_peak,df
        else:
            width  = cap.get(3) # Pixel width of video
            height = cap.get(4) # Pixel height of video
            fs = cap.get(5) # Sampling rate of video
        for pp,peak in enumerate(peaks):
            fig,ax=plt.subplots(1,4,figsize=(15,5))    
            frames=[start_peak[pp], peak, stop_peak[pp]]
            if any(np.array(frames)<0):
                print('Negative frame requesteds')
            parts=['head_centroid','rear_centroid']
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
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(rear_dir + '/rear_%03d.png' % pp )
            plt.close()
        cap.release()
    return peaks,start_peak,stop_peak,df

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
                peak,start,stop,df = detect_rear(path,rear_thresh=0.7,min_thresh=0.2,save_figs=False,
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
                peak_array,start,stop,df = detect_rear(path,rear_thresh=0.7,min_thresh=0.2,save_figs=False,
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