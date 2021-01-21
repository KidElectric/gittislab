from gittislab import signal, behavior, dataloc, ethovision_tools, plots
import os
from pathlib import Path
import numpy as np
import pandas as pd
# import modin.pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem, t
import pdb
import math
import time


def batch_analyze(basepath,inc,exc):
    '''
    With comments
    '''
    data=pd.DataFrame([],columns=['anid','proto','cell_area_opsin',
                                  'amb_vel','amb_meander','amb_bouts','amb_directed'])
    temp=data
    min_bout=1
    use_dlc=False
    use_cols=['time','vel','im','dir','ambulation','meander']
    for ii,ee in zip(inc,exc):
        pns=dataloc.raw_csv(basepath,ii,ee)
        for pn in pns:
            temp={}
            raw,meta=ethovision_tools.csv_load(pn,columns=use_cols,method='preproc' )
            temp['anid']=meta['anid'][0]
            temp['cell_area_opsin']='%s_%s_%s' % (meta['cell_type'][0],
                                                     meta['stim_area'][0],
                                                     meta['opsin_type'][0])
            temp['proto']=meta['protocol'][0]
            stim_dur = round(np.mean(meta['stim_dur']))        
            vel_clip=behavior.stim_clip_grab(raw,meta,
                                              y_col='vel', 
                                              stim_dur=stim_dur)        
            clip_ave=behavior.stim_clip_average(vel_clip)   
            
            #### Calculate stim-triggered %time mobile:
            percentage = lambda x: (np.nansum(x)/len(x))*100
            raw['m']=~raw['im']
            m_clip=behavior.stim_clip_grab(raw,meta,y_col='m', 
                                            stim_dur=stim_dur, 
                                            summarization_fun=percentage)        
            
            #### Calculate ambulation bout properties:
            raw['run'] = (raw['ambulation']==True) & (raw['vel']>5)
            # raw['flight']=(raw['vel'] > (4* np.mean(raw['vel']))) #Flight, Yilmaz & Meister 2013
            raw['flight']=(raw['vel'] > (3 * np.mean(raw['vel'])))
            if any(raw['run']):
                amb_bouts=behavior.bout_analyze(raw,meta,'flight',
                                                stim_dur=stim_dur,
                                                min_bout_dur_s=min_bout,
                                                use_dlc=use_dlc)
                temp['amb_meander']=np.nanmean(amb_bouts['meander'],axis=0)
                temp['amb_directed']=np.nanmean(amb_bouts['directed'],axis=0)
                temp['amb_bouts']=np.nanmean(amb_bouts['rate'],axis=0)
            else:
                temp['amb_meander']=[np.nan, np.nan, np.nan]
                temp['amb_directed']=[np.nan, np.nan, np.nan]
                temp['amb_bouts']=[0,0,0]
            #### Calculate immobile bout properties:
            im_bouts=behavior.bout_analyze(raw,meta,'im',
                                            stim_dur=stim_dur,
                                            min_bout_dur_s=min_bout,
                                            use_dlc=use_dlc)
    
            data=data.append(temp,ignore_index=True)