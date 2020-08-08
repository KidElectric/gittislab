def detect_rear(dlc_h5_path,rear_thresh=0.65,min_thresh=0.25,save_figs=False,
                dlc_outlier_thresh_sd=4,dlc_likelihood_thresh=0.1):    
    import os
    from gittislab import signal # sys.path.append('/home/brian/Dropbox/Python')
    from gittislab import dataloc
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    import cv2
    from scipy.interpolate import interp1d
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
                same as above except 
    '''
    from gittislab import dataloc
    from gittislab import mat_file
    import numpy as np
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

def prob_rear(is_rear,laserOn,window):
    print('Empty')