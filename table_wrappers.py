#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:10:27 2021

@author: brian
"""
import pandas as pd
import numpy as np

def consolidate_columns_to_labels(df,label_columns=[],
                                  value_column_name='value',
                                  label_column_name='label'):
    ''' Take a dataframe with several columns and convert those column names
    to label values, and put values in same row of value column.
    
    e.g. if df = 'a' 'b' 'c' 'id'
                  1   2   3  'apple'
                  3   4   5  'dog'
                  
         return  'label' 'value' 'id'
                   a      1    'apple'
                   b      2    'apple'
                   c      3    'apple'
                   a      3    'dog'
                   b      4    'dog'
                   c      5    'dog'
                   
    '''
    
    #Initialize the new dataframe in format desired:
    keep_columns = [col for col in df.columns if (col not in label_columns)]
    output_df={value_column_name : [],
            label_column_name : [],
            }
    for col in keep_columns:
        output_df[col]=[]
    
    
    #Turn dataframe into a column of labels and values, w/ chunks ID'd by index:
    unwrapped =df.stack().reset_index(level=1, name='val')
    ind = np.unique(df.index)
    
    for i in ind:
        #For each chunk of rows in unwrapped data frame        
        chunk = unwrapped.loc[i].reset_index(drop=True)
        labels = chunk.loc[:,'level_1']
        
        #Keep label/value pairs as is, but add in
        for ii,lab in enumerate(labels):
            if lab in label_columns:
                output_df[value_column_name].append(chunk.loc[ii,'val'])
                output_df[label_column_name].append(lab)
            else:
                for j in range(0,len(label_columns)):
                    output_df[lab].append(chunk.iloc[ii,1])
    return pd.DataFrame(output_df)