#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:03:08 2020
Test creating plot analysis of 10x10 / 10x30 type data
@author: brian
"""

# %%
from gittislab import signal, behavior, dataloc, ethovision_tools, plots
import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import sem, t

# inc=['GPi','CAG','Arch','10x30','AG6151_3_CS090720']
inc=['AG','GPe','FoxP2','ChR2','10x10_20mW',]
exc=['exclude','_and_Str','Left','Right']
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.rawh5(basepath,inc,exc)
raw_df,raw_par=ethovision_tools.h5_load(pns[0])
clip=behavior.stim_clip_grab(raw_df,raw_par,y_col='vel')
clip_ave=behavior.stim_clip_average(clip)
fig,ax = plots.mean_disc_plus_conf(clip,['Pre','Dur','Post'])
plots.mean_cont_plus_conf(clip_ave,xlim=[-10,20],highlight=[0,10,20])