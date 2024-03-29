import math
import numpy as np
import pdb
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.stats import t
import librosa 

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    if not isinstance(data,pd.core.series.Series):
        data=pd.core.series.Series(data)
    data.fillna(method='ffill',inplace=True)
    data.fillna(method='bfill',inplace=True)
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data.values)
    return y

def pad_lowpass_unpad(data,cutoff,fs,order=5):
    pad=int(round(fs*2))
    
    #Remove outliers
    existing_nan=np.isnan(data)
    data=outlier_to_nan(data,outlier_thresh=3.5)
    data[existing_nan]=np.nan
   
    #Fill in nans if pandas series:
    if not isinstance(data,pd.core.series.Series):
        data=pd.core.series.Series(data)
        
    data=data.interpolate(method='pad').astype(np.float)
    
    #Pad:
    data=np.pad(data,pad_width=(pad,),mode='linear_ramp')
    
    #Low-pass filter:     
    data=butter_lowpass_filtfilt(data, cutoff, fs, order=order)
    
    #Return unpadded:
    return data[pad:-pad]

def conf_int_on_matrix(y,axis=1,conf=0.95):
    #Return mean +/- 95% Confidence interval on a matrix
    n=y.shape[axis]
    ym=np.nanmean(y,axis=axis)
    std_err = np.nanstd(y,axis=axis)/np.sqrt(n)
    h = std_err * t.ppf((1 + conf) / 2, n - 1)
    y_conf_int=np.array([ym-h, ym+h]).T
    return ym, y_conf_int

def outlier_to_nan(y,outlier_thresh=3):
    if not isinstance(y,pd.core.series.Series):
        y=pd.core.series.Series(y)
        
    y=y.fillna(method='ffill').astype(np.float)
    y=y.fillna(method='bfill').astype(np.float)
    y=np.array(y.values.astype(float))
    
    # y[np.isnan(y)]=0
    dy=np.diff(np.concatenate(([y[0]],y)))
    on, _ = thresh(dy, outlier_thresh,'Pos') # Find onset of outlier
    _, off =thresh(dy, outlier_thresh,'Neg')
    if (len(on) > 0) and (len(off) >0):
        for i,j in zip(on,off):
            if j < i:
                ii=j
                j=i
                i=ii
                i = i-2       
            #Expand:
            i = i-2
            j = j+2
    
            y[i:j]=np.nan
    return y

def thresh(y,thresh, sign='Pos'):
    '''
    

    Parameters
    ----------
    y : np.array
       array of signal to threshold
    thresh : numeric (int / float)
        Value to use in thresholding
    sign : String, optional
        Whether to return periods above ('Pos') or below ('Neg') the threshold.
        The default is 'Pos'.

    Returns
    -------
    onsets -
    np.array
        Array of index locations in y where threshold crossing begins.
    
    offsets
        np.array
        Array of index locations in y where threshold crossing ends.

    '''
    #Fill in nans using pandas:
    if not isinstance(y,pd.core.series.Series):
        y=pd.core.series.Series(y.flatten())
    

    y=y.fillna(method='ffill').astype(np.float)
    y=y.fillna(method='bfill').astype(np.float)
    y=np.array(y)

    if sign == 'Pos':
        def eval_thresh(v,thresh):
            return v > thresh
    elif sign == 'Neg':
        def eval_thresh(v,thresh):
            return v < thresh
    onsets=[]
    i=-1
    offsets=[]
    while i <  (len(y)-1):
        i+=1
        v=y[i]        
        if eval_thresh(v,thresh) == True:
            onsets.append(i)
            while (eval_thresh(v,thresh)==True) and (i < (len(y)-1)):
                i=i+1     
                v=y[i] 
            offsets.append(i)
    return onsets, offsets

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    
    FROM : https://scipy-cookbook.readthedocs.io/items/idx_signal_processing.html 
    """

    if x.ndim != 1: 
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    # pdb.set_trace()
    y=np.convolve(w/w.sum(),s,mode='valid')
    start = int((window_len-1)//2)
    stop = int(-(window_len-1)//2)
    return y[start:stop]

def bin_analyze(x,y,bin_dur,fun = np.mean):
    '''
    

    Parameters
    ----------
    x : 1D np.array of time
    y : 1D np.array of varible to bin
    bin_dur : Bin size in units of time used in x
    fun : function to perform on each bin, Default: np.mean()

    Returns
    -------
    bin_x : np.array of binned x
    bin_out : np.array of binned y


    '''    
    s=0
    bin_out=[]
    bin_x = []
    x=np.array(x)
    y=np.array(y)
    while s < x[-1]:
        ind = (x > s) & (x < (s + bin_dur))
        bin_out.append(fun(y[ind]))
        bin_x.append(s)
        s += bin_dur
    return np.array(bin_x),np.array(bin_out)

# def 2d_histogram(x,y,bins,fun=np.nansum):
#     see: np.histogram2d(x, y, bins=10, range=None, normed=None, weights=None, density=None)    
#     return xy

def chunk_by_x(x,y,x_points,x_range):
    '''
        Take x y arrays and make a matrix of data clips from y, centered on x_points, 
        and extending x_range [-xx +xx]. 
        number of clips = len(x_points) = rows of output
        length of clips = x_range[1]-xrange[0] = columns of output (based on sampling rate)
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    x_points : TYPE
        DESCRIPTION.
    chunk_range : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    output=[]
    fs=1/np.nanmean(np.diff(x))
    
    if x_range[0] > 0:
        x_range[0] = -1*x_range[0]
    for p in x_points:
        if (p + x_range[0]) < np.min(x):
            #ideally nan pad the beginning of clip, for now exclude
            #output.append(np.nan)
            print('Missing beginning of a clip... skipping')
        elif (p + x_range[1]) > np.max(x):
            #nan pad the end of clip, for now exclude
            #output.append(np.nan)
            print('Missing end of a clip... skipping')
        else:
            #take the clip
            use_x= (x >= (p +x_range[0] )) & (x < (p +x_range[1] ))
            clip=y[use_x]
            output.append(clip)
    return output

def get_spectral_band_power(y,fs,low,high):
    if not isinstance(y,pd.core.series.Series):
        y=pd.core.series.Series(y.flatten())
    y.fillna(method='ffill',inplace=True)
    y.fillna(method='bfill',inplace=True)
    n_fft = 256
    hop_length=round(fs/3) 
    freqs = np.arange(0, 1 + n_fft / 2) * fs / n_fft
    S = librosa.feature.melspectrogram(y=y.values, sr=fs, n_fft= n_fft, hop_length=hop_length)
    ind=(freqs[0:-1] > low) & (freqs[0:-1] < high)
    dm= np.mean(S[ind,:],axis=0)
    
    #resample to original sampling rate of y:
    i=0
    out=np.ones(y.shape)
    for h in dm:
        out[i:(i+hop_length)]=h
        i=i+hop_length
    return out

def boxcar_smooth(y,samps):
    '''
    boxcar_smooth(y,samps)     
    Perform a padded 1d sliding average smooth via convolution.
    
    Inputs: y, array - data to smooth
            samps, int - window for convolution in samples
    
    Output: y_smooth, array - smoothed y data
    
    '''
    pad=np.ones(samps)
    pad0=pad*y[0]
    padN=pad*y[-1]
    y_smooth=np.concatenate((pad0,y,padN))
    
    box = np.ones(samps)/samps
    y_smooth = np.convolve(y_smooth, box, mode='same')
    return y_smooth[samps:-samps]


def gaussian_filter1d(size,sigma):
    filter_range = np.linspace(-int(size/2),int(size/2),size)
    gaussian_filter = [1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-x**2/(2*sigma**2)) for x in filter_range]
    return gaussian_filter


def join_gaps(on,off,min_samp):
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
                while last_i < (len(off)-1) and (on[last_i+1] - off[last_i]) < min_samp:
                    last_i +=1
                # if last_i >= len(off):
                #     last_i=len(off)-1
                keep_off.append(last_i)
    return on[keep_on],off[keep_off]

def angle_delta(b1, b2):
    r=(b2 - b1) % 360.0
    if r >= 180.0:
        r -= 360.0
    return r

def angle_vector_delta(b1, b2,thresh=None,fs=29.97):
    out=[]
    for a,b in zip(b1,b2):
        out.append(angle_delta(a,b))
    out=np.abs(np.array(out))
    if thresh != None:
        out[out >thresh]=np.nan
        out=pd.core.series.Series(out)
        cutoff=3 #Hz
        out= pad_lowpass_unpad(out,cutoff,fs,order=5)
    return out

def ismember(a, b):
     B_unique_sorted = np.unique(b)
     B_in_A_bool = np.in1d(B_unique_sorted, a, assume_unique=True)
     return B_in_A_bool
    # bind = {}
    # for i, elt in enumerate(b):
    #     if elt not in bind:
    #         bind[elt] = i
    # return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

def one_line_angle(x1,y1,x2,y2):
    return math.degrees(math.atan2(y2-y1, x2-x1))

def two_line_angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))

def expand_peak_start_stop(y,distance=30,height=0.6,width=10, min_thresh=0.2):
    # Use find_peaks() and generate boundaries additionally defined by a minimum threshold
    peaks=find_peaks(y,distance=distance,height=height,width=width)[0] #Will return np.array
    start_peak=[]
    stop_peak=[]
    rem_peaks=[]
    #Expand peak boundaries to satisfy minimum threshold (min_thresh):
    for i,peak in enumerate(peaks):   
        y_loc=y[peak]
        n=0
        while y_loc > min_thresh and (peak-n) > 0:
            y_loc=y[peak-n]
            n+=1
        if i>0 and start_peak[-1] == peak-(n-1):
            rem_peaks.append(i)
        else: #Add start and stop index to lists to keep:
            start_peak.append(peak-(n-1))
            n=0
            y_loc=y[peak]
            while y_loc > min_thresh and (peak+n) < len(y):
                y_loc=y[peak+n]
                n+=1
            stop_peak.append(peak+(n-1))
    peaks=np.delete(peaks,rem_peaks) #Delete repeat peak- detections after expanded boundaries
    
    return peaks,np.array(start_peak),np.array(stop_peak)

def calculateDistance(x1,y1,x2,y2): 
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist 

def log_modulus(x):
    '''
    Perform the log modulus transform of x (sign(x) * log(|x| + 1))
    '''
    
    return np.sign(x) * np.log10(np.abs(x)+1)

def scale_per_dist(x,head_xy,tail_xy,mouse_height,step=2,poly_order = 2):
    x = x - np.nanmin(x)
    max_dist=math.ceil(np.nanmax(x))  
    xtemp=np.array([i for i in range(0,max_dist,step)]) + step / 2
    keep_x = []
    out = []
    for i, dist in enumerate(range(0, max_dist, step)):
        subx=(x>dist) & (x< (dist + step))        
        if any(subx):
            mouse_length= np.nanmax(abs(head_xy[subx,0] - tail_xy[subx,0]))
            out.append(mouse_length) #Take max value of each bin of x
            keep_x.append(xtemp[i])
    #Remove nan:
    out= np.array(out).flatten()
    keep_x=np.array(keep_x)
    ind = np.isnan(out) == False
    keep_x=keep_x[ind]
    out=out[ind]
    # pdb.set_trace()
    p=np.poly1d(np.polyfit(keep_x,out,poly_order))
    scale = p(x)/p(0)
    return mouse_height * scale

def max_normalize_per_dist(x,y,step=2,poly_order=2):
    '''
    Normalize local maximum values of y as a (polynomial) function of x.
    Take max value of each bin of x, fit a polynomial, and divide y by the fitted max
    This is useful to correct object size as a function of distance from a camera, for example.
    ''' 
    
    max_val=math.ceil(np.nanmax(x))
    out=[]
    xtemp=np.array([i for i in range(0,max_val,step)])+step/2
    keep_x=[]
    for i,ind in enumerate(range(0,max_val,step)):
        subx=(x>=ind) & (x< (ind + step))
        suby=y[subx]
        if any(suby):
            out.append(np.nanmax(suby)) #Take max value of each bin of x
            keep_x.append(xtemp[i])
    
    #Remove nan:
    out= np.array(out).flatten()
    keep_x=np.array(keep_x)
    ind = np.isnan(out) == False
    keep_x=keep_x[ind]
    out=out[ind]
    
    p=np.poly1d(np.polyfit(keep_x,out,poly_order))
    # pdb.set_trace()
    norm_factor = p(x)
    if np.mean(norm_factor) < 25:
        # Suspect no rears in video
        norm_factor = norm_factor * 3
    
        
    norm_height = y / norm_factor
    return norm_height