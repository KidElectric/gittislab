#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:27:21 2021

@author: brian
"""


from gittislab import signals, behavior, dataloc, ethovision_tools, plots, table_wrappers
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import pdb
from itertools import compress
import seaborn as sns

ex0=['exclude','Bad','GPe','bad','Broken', 'grooming','Exclude','Other XLS']
inc=[['AG','hm4di','Str','A2A','Ai32']]
# inc=[['AG','Str','D2_D1','ChR2_hM3Dq','10x10_15mW']]
make_preproc = False
exc=[ex0]
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
        
    basepath = 'F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\'
else:
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

# %% Plot 15min free running mouse summary:
ex0=['10hz','exclude','Bad','bad',
     'Broken', 'grooming','Exclude','Other XLS']
exc=[ex0]
conds = ['saline','cno']
keep={}
for cond in conds:
    inc=[['AG','hm4di','Str','A2A','Ai32','15min',cond]]
    data = behavior.free_running_summary_collect(basepath,inc,exc)
    keep[cond]=data
    # plots.plot_freerunning_mouse_summary(data,)

# %% Plot both conditions:
plots.plot_freerunning_cond_comparison(keep,save=False,close=False)

