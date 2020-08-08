#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:37:06 2020

@author: brian
"""

def get_subs(axes):
    import numpy as np
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
    from matplotlib.pyplot import gca 
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
    import cv2
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
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
    import numpy as np
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