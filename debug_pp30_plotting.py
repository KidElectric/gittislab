#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:04:21 2020

@author: brian
"""

from gittislab import signal, behavior, dataloc, ethovision_tools, plots
import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# %% Fix bugs 1 by 1..... and repeat
# Make .h5
inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim',]]
exc=[['exclude','_and_Str','Left','Right']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
xlsx_paths=dataloc.rawxlsx(basepath,inc[0],exc[0])
raw,params=ethovision_tools.raw_params_from_xlsx(xlsx_paths[1])

# v=behavior.smooth_vel(raw,params,win=10)
# %%
tt=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/pp30_cond_dish_fc_stim/AG4486_3_KA062119/Raw_AG4486_3_KA062119.h5') #File can not currently be read back in
test_file=dataloc.rawxlsx(tt.parent)
raw,params=ethovision_tools.raw_params_from_xlsx(test_file)
# %% Test save as h5
pn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/pp30_cond_dish_fc_stim/AG4486_3_KA062119/Raw_AG4486_3_KA062119.h5')
ethovision_tools.h5_store(pn,raw)
# %% Load
store = pd.HDFStore(pn)
data = store['mydata']
# %% Finally this should work (pp30-> WORKS!)
inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim']]
exc=[['exclude','_and_Str','Left','Right']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'

ethovision_tools.unify_to_csv(basepath,inc,exc,force_replace=True)

# %% Load example data
inc=[['AG','GPe','CAG','Arch','pp30_cond_dish_fc_stim',]]
exc=[['exclude','_and_Str','Left','Right']]
basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
pns=dataloc.rawh5(basepath,inc[0],exc[0])
raw_df,raw_par=ethovision_tools.h5_load(pns[0])


# %%
pn=Path('/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/pp30_cond_dish_fc_stim/AG4486_3_KA062119/Raw_AG4486_3_KA062119.h5')
raw_df,raw_par=ethovision_tools.h5_load(pn)

# %%
df=pd.DataFrame()
for key in params.keys():
    df.loc[key,0]=params[key]