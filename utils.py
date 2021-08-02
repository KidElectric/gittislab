#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 11:54:56 2021

@author: brian
"""
import colorama 
import numpy as np

def warning(message):
    #Use colorama to print a warning (not yet implemented)
    print('colorama')

def print_percent(num_ind,denom_ind,lab_1 = 'Index 1',lab_2='Index 2'):  
    num=np.sum(num_ind)
    den=np.sum(denom_ind)
    percent=num/den * 100
    out='%s is %3.1f%% of %s (%d/%d)' % (lab_1,percent,lab_2,num,den)
    print(out)
    return out
    