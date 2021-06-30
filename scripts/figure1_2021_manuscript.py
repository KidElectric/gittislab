#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 16:25:54 2021

@author: brian
"""


from gittislab import signals, behavior, dataloc, ethovision_tools, plots, model
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib import pyplot as plt
import pdb
from itertools import compress


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
    
# %% A2a ChR2
ex0=['exclude','Bad','GPe','bad',\
     'Broken','15min','10hz', 'Exclude','Other XLS','AG3233_5','AG3233_4',\
         'AG3488_7']
inc=[['AG','Str','A2A','ChR2','10x30','Bilateral'],['AG','Str','A2A','Ai32','10x30','Bilateral']]
exc=[ex0,ex0]
data = behavior.open_loop_summary_collect(basepath,inc,exc)
fig=plots.plot_openloop_mouse_summary(data)

# %% CAG Arch GPe
ex0=['exclude','Bad','Str','bad',\
     'Broken','15min','10hz', 'Exclude','Other XLS']
inc=[['AG','GPe','CAG','Arch','10x30',]]
exc=[ex0]
data = behavior.open_loop_summary_collect(basepath,inc,exc)
fig=plots.plot_openloop_mouse_summary(data)
dest = '/home/brian/Desktop/test.pdf'
plt.savefig(dest, dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)