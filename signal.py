import math
import numpy as np
from scipy.signal import find_peaks 
from scipy.signal import butter, filtfilt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def thresh(y,thresh, sign='Pos'):
    if len(y.shape) == 1:
        y=y[:,None]
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
    if onsets.size > offsets.size:
        onsets=onsets[0:offsets.size]
    elif offsets.size > onsets.size:
        dif= offsets.size - onsets.size
        offsets=offsets[dif:offsets.size]
        print('test')
    if sign == 'Neg':
        y *= -1
    return onsets, offsets

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

def join_crossings(on,off,min_samp):
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

def peak_start_stop(y,distance=30,height=0.6,width=10, min_thresh=0.2):
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
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
    return dist 


def max_correct(x,y,step,poly_order=2):
    # Correct local maximum values of y as a (polynomial) function of x.
    # Take max value of each bin of x, fit a polynomial, and divide y by the fitted max
    # This is useful to correct object size as a function of distance from a camera, for example.
    max_val=math.ceil(max(x))
    out=[]
    xtemp=np.array([i for i in range(0,max_val,step)])+step/2
    keep_x=[]
    for i,ind in enumerate(range(0,max_val,step)):
        subx=np.argwhere((x>=ind) & (x< (ind + step)))
        suby=y[subx]
        if any(suby):
            out.append(max(suby)) #Take max value of each bin of x
            keep_x.append(xtemp[i])
    p=np.poly1d(np.polyfit(keep_x,np.array(out)[:,0],poly_order))
    return y/p(x)