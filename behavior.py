from gittislab import dataloc, mat_file, signal, ethovision_tools
import numpy as np
import os
import math
from matplotlib import pyplot as plt
import pandas as pd
import cv2
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

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
    
    #Low-pass filter method:
    x=raw['x'].values
    y=raw['y'].values
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
    for i,x in enumerate(x_s):
        x2=x
        x1=x_s[i-1]
        y1=y_s[i-1]
        y2=y_s[i]
        dist=signal.calculateDistance(x1,y1,x2,y2)
        vel.append(dist / (1/fs))
    return np.array(vel)

def measure_bearing(raw_df,raw_par):
    # Measurement revolves around direction of travel when crossing midline of open field
    # and subsequent continuation along same route or rebounding
    return raw_df['time'].values[0]

def stim_clip_grab(raw_df,raw_par,raw_col,baseline=10,stim_time=10,summarization_fun=np.nanmean):
    
    fs=raw_par['fs']
    nsamps=math.ceil(((baseline*2) + stim_time) * fs)
    ntrials=len(raw_par['stim_on'])
    cont_array=np.empty((nsamps,ntrials))
    cont_array[:]=np.nan
    data=raw_df[raw_col].values

    disc_array=np.empty((ntrials,3)) #Pre Dur Post
    for i,on in enumerate(raw_par['stim_on']):
        # Array containing continuous part of analysis:
        off=raw_par['stim_off'][i]
        base_samp= round((on-baseline) * fs)
        on_samp = round(on * fs)
        stim_time_samp= round((on+stim_time) * fs)
        off_samp = round(off * fs)
        post_samp = round((off+baseline) * fs)
        intervals=[[base_samp,on_samp],
                   [on_samp,stim_time_samp],
                   [off_samp,post_samp]]
        
        cont_array[:,i]=data[base_samp:(base_samp + nsamps)]
        
        # (1,3) Array containing Pre, Dur, Post discretized analysis:
        for ii,interval in enumerate(intervals):
            disc_array[i][ii]=summarization_fun(data[interval[0]:interval[1]])

    out_struct={'cont':cont_array,'disc':disc_array,'samp_int':intervals}
    return out_struct
    
def mouse_stim_vel(raw_df,raw_par,baseline=10,stim_time=10):
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
    out_struct=stim_clip_grab(raw_df,raw_par,raw_col,baseline=10,stim_time=10,summarization_fun=np.nanmedian)
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