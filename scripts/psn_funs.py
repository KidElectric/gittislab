#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:34:19 2021

@author: brian
"""
import pandas as pd
from pathlib import Path
import numpy as np
import pdb 
lab_dict = {'Rock':1, 'Paper':2, 'Scissor':3}
def boris_tabular_to_full(pn):
    pn = Path(pn)
    dat=pd.read_csv(pn, header=15, sep='\t')
    dur=dat.loc[0,'Total length']
    fps=dat.loc[0,'FPS']
    nframes= int(( dur*fps) //1)
    label=np.zeros((nframes,1))
    time = np.arange(0,nframes)/fps
    start_time=0
    stop_time=0
    for index,row in dat.iterrows():
        if row['Status'] == 'START':
            start_time = row['Time']
        else:
            stop_time = row['Time']
            ind = np.argwhere((time >= start_time ) & (time < stop_time))
            label[ind] = lab_dict[row['Behavior']]
    new_file = pn.parts[-1].split('.')[0] + '_full.csv'
    out=pd.DataFrame(data=time[:,np.newaxis], columns=['time'])
    out['labels']=label
    out.to_csv(pn.parent.joinpath(new_file))
    return label


pns=['/home/brian/Dropbox/DataTransfer/dual_cam_dlc/boris/bi_left_tabular.tsv',
     '/home/brian/Dropbox/DataTransfer/dual_cam_dlc/boris/bi_right_tabular.tsv']
for pn in pns:
    lab=boris_tabular_to_full(pn)