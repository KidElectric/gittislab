#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:13:01 2021

@author: brian
"""
import os

# %%

# basepath='/home/brian/media/gittislabserver/Str'

basepath= '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
search_string = 'AG4668_'
for dirpath, _, _ in os.walk(basepath,followlinks=True):
    if search_string in dirpath:
        print(dirpath)

