#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:10:27 2021

@author: brian
"""
import pandas as pd
import numpy as np
import pdb

def consolidate_columns_to_labels(df,label_columns,
                                  value_column_name='value',
                                  label_column_name='label'):
    ''' Take a dataframe with several columns and convert those column names
    to label values, and put values in same row of value column.
    Update: this seems to be the same as pd.melt() !!!!
    
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


def df_melt_stack(dfs,df_conds,label_columns, var_name, value_name,                        
                  static_columns, sort_column=None):
    ''' Take 2 dataframes of different conditions, melt columns and concatenate.
        Input: 
        dfs = list of pandas.DataFrames with identical column variables,
        df_conds = list of string labels describing condition of data frames ['control','experimental']
        label_columns = list of column names to unpivot (pd.melt(value_vars))
        static_columns = list of column names to fill when unpivotted  (pd.melt(id_vars))
        sort_column (optional) = string. name of column to use for sorting first if desired.
        
        Output:
            df_out= pandas dataframe with melted,stacked data.
            
        e.g. 
        
        if dfs[0] = 'a' 'b' 'c' 'id'
                     1   2   3  'apple'
                     3   4   5  'dog'
                     
           dfs[1] = 'a' 'b' 'c' 'id'
                     0   0   0  'apple'
                     1   3   2  'dog'
                     
             with df_conds=['saline','cno']
                  static_columns=['id']
                  
         return:
                 'var_name' 'value_name' 'id'        'cond'
                   a            1         'apple'   'saline'
                   b            2         'apple'   'saline'
                   c            3         'apple'   'saline'
                   a            3         'dog'     'saline'
                   b            4         'dog'     'saline'
                   c            5         'dog'     'saline'
                   a            0         'apple'   'cno'
                   b            0         'apple'   'cno'
                   c            0         'apple'   'cno'
                   a            0         'dog'     'cno'
                   b            3         'dog'     'cno'
                   c            2         'dog'     'cno'
                   
            
    '''
    static_columns += ['cond']
    df_out = pd.DataFrame()
    for cond_label,df in zip(df_conds,dfs):
        df['cond']=cond_label
        # pdb.set_trace()
        df_out= pd.concat((df_out,
            pd.melt(df,
                    value_vars = label_columns,
                    id_vars = static_columns, 
                    value_name=value_name,
                    var_name=var_name)
            ))
    return df_out