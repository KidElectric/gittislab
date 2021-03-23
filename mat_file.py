from scipy.io import loadmat
import numpy as np
import os
from gittislab import dataloc
import pandas as pd


def load(path,verbose=True):
    '''
    mat_file.load() - a wrapper for scipy.io.loadmat()

    Parameters
    ----------
    path : Path
        File path of .mat file
    verbose: Boolean
        Whether to print string describing file location (verbose=True, default)

    Returns
    -------
    mat : dict
        default .mat import dictionary of file. For nested structures, use dtype_array_to_dict()

    '''
    
    if verbose == True:
        path_str=dataloc.path_to_description(path)
        print('\tLoading ~%s.mat...' % path_str)
    mat=loadmat(path)
    if verbose == True:
        print('\t\tFinished')
    return mat

def array_of_arrays_to_flat_df(dfa):
    if not isinstance(dfa,pd.DataFrame):
        dfa=pd.DataFrame(dfa)
    column_headers=[x[0] for x in dfa.loc[0,:]]
    new_df={}
    for i,col in enumerate(column_headers):
        test_val=dfa.loc[1,i]
        if len(test_val) ==1:
            if len(test_val[0])==1:
                values=[x[0][0] for x in dfa.loc[1:,i]]
            else:
                values=[x.flatten() for x in dfa.loc[1:,i]]
        else:
            values=[x.flatten() for x in dfa.loc[1:,i]]
        new_df[col]=values
    return pd.DataFrame(new_df)

def dtype_array_to_dict(mat,field):
    '''
    mat_file.dtype_array_to_dict() - takes arrays with dtype descriptions in a
    given field ('field') of mat file dictionary, and makes a dictionary.

    Parameters
    ----------
    mat : dict
        Default .mat file after import using loadmat().
    field : str
        String describing field to use in mat file dict:
            e.g. if field = 'params', clean up mat['params']

    Returns
    -------
    new_dict : dict
        Dictionary where each dtype.name of field in mat is used to create
        dictionary keys with corresponding array values.

    '''
    fields=mat[field].dtype.names
    new_dict={}
    for i,f in enumerate(fields):
        temp=mat[field][0][0][i]
        if temp.size > 0:
            new_dict[f]=temp[0]
        else:
            new_dict[f]=np.nan
    return new_dict