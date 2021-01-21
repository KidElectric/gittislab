import pandas as pd
# import modin.pandas as pd
from gittislab import dataloc, signal, behavior, mat_file
import os #To help us check if a file exists
from pathlib import Path
import numpy as np
import math
import yaml


if os.name == 'posix':
    sep='/'
else:
    sep='\\'
def unify_to_h5(basepath,conds_inc=[],conds_exc=[],force_replace=False,win=10):
    '''
    unify_to_h5(basepath,conds_inc=[],conds_exc=[]):
            makes a fully integrated .h5 file containing:
            a) meta - dictionary with key experiment information (animal id, condition, etc).
            b) raw - dataframe with columns of samples for various measurements like mouse position (x,y)
            c) stim - dictionary with stimulation times and type (if relevant)
            d) deeplabcut - dataframe of body tracking if available

    Parameters
    ----------
    basepath : TYPE
        DESCRIPTION.
    conds_inc : TYPE, optional
        DESCRIPTION. The default is [].
    conds_exc : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    None.

    '''
    
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        xlsx_paths=dataloc.rawxlsx(basepath,inc,exc)
        if isinstance(xlsx_paths,Path):
            xlsx_paths=[xlsx_paths]
        for ii,path in enumerate(xlsx_paths):
            # First, let's check if there is already a .h5 file in this folder:    
            new_file_name=dataloc.path_to_rawfn(path) + '.h5'
            file_exists=os.path.exists(path.parent.joinpath(new_file_name))
            print('Inc[%d], file %d) %s generation...' % (i,ii,new_file_name))
            if (file_exists == False) or ((file_exists == True) and (force_replace == True)): 
                matpath=dataloc.rawmat(path.parent)
                xlsxpath=path
                if matpath: #If .mat file exists, create raw .h5 file from this file
                    raw=raw_from_mat(matpath) #Will add 'raw', 'meta', and 'stim' to .h5
                    meta=meta_from_mat(matpath)
                else: #If no .mat file exists, generate raw .h5 from xlsx (slow)
                    print('\tNo .mat file found! Generating from .xlsx...')
                    raw,meta=raw_meta_from_xlsx(xlsxpath)

                
                #Set common boolean columns (but don't force it if these columns are not present):
                common_bool=['im', 'm', 'laserOn','iz1','iz2']
                for cb in common_bool:
                    if cb in raw.columns:
                        raw[cb]=raw[cb].astype('bool')
                
                #Improved velocity measure:
                
                raw['vel']=behavior.smooth_vel(raw,meta,win)
                meta['vel_smooth_win_ms']=win/meta['fs'] * 1000 # ~333ms
                
                thresh=2; #cm/s; Kravitz & Kreitzer 2010
                dur=0.5; #s; Kravitz & Kreitzer 2010
                raw=add_amb_to_raw(raw,meta,thresh,dur,im_thresh=1,im_dur=0.25) #Thresh and dur
                meta['amb_vel_thresh']=thresh
                meta['amb_dur_criterion_ms']=dur
                
                #Assume a Raw*.h5 now exists and add deeplabcut tracking to this .h5:
                dlcpath=dataloc.dlc_h5(path.parent)
                
                #add_deeplabcut(path) # still need to write this function -> integrated into raw? something else?
                
                #Write raw and meta to .h5 file:
                pnfn=path.parent.joinpath(new_file_name)
                print('\tSaving %s\n' % pnfn)
                h5_store(pnfn,raw,**meta)
                
            else:
                # path_str=dataloc.path_to_description(path)
                print('\t %s already EXISTS in %s.\n' % (new_file_name,path.parent))
    print('Finished')

def add_dlc_to_csv(basepath,conds_inc=[[]],conds_exc=[[]],
                   save=False,force_replace = False):
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        csv_paths=dataloc.raw_csv(basepath,inc,exc)
        if isinstance(csv_paths,Path):
            csv_paths=[csv_paths]
        for ii,path in enumerate(csv_paths):            
            # First, let's check if there is a dlc .h5 file in this folder:                 
            dlc_path=dataloc.gen_paths_recurse(path.parent,inc,exc,'*dlc_analyze.h5')
            if isinstance(dlc_path,Path):
                file_exists = True
            else:
                file_exists = len(dlc_path) > 0

            if (file_exists == True):       
                if save == True:
                    print('Inc[%d], file %d) %s adding to Raw*.csv ...' % (i,ii,dlc_path))
                #Read raw and meta to .csv files:               
                print('\tLoading %s\n' % path)
                raw,meta= csv_load(path)
                
                #Check if DLC already added:
                dlc_added = 'dlc_side_tail_base_x' in raw.columns
                #If not, or if set to force replace:
                if (dlc_added == False) or ((dlc_added==True) and (force_replace==True)):
                    # Read in the DLC file:
                    print('\tLoading DLC file...')
                    dlc_outlier_thresh_sd=4
                    dlc_likelihood_thresh=0.1
                    dlc= behavior.load_and_clean_dlc_h5(dlc_path,dlc_outlier_thresh_sd,
                                                        dlc_likelihood_thresh)
                    meta['dlc_outlier_thresh_sd'] = dlc_outlier_thresh_sd
                    meta['dlc_likelihood_thresh'] = dlc_likelihood_thresh
                    
                    #Add rearing to dlc:
                    rear_thresh=0.65
                    min_thresh=0.25
                    _,_,_,dlc = behavior.detect_rear(dlc,rear_thresh,min_thresh)
                    meta['rear_thresh']=rear_thresh
                    meta['rear_min_thresh']=min_thresh
                    
                    # Make sure locomotion state fields are up-to-date:
                    raw = add_amb_to_raw(raw,meta)
                    
                    # For each column in dlc, make a column in raw with collapsed name:
                    for col in dlc.columns:                    
                        if 'likelihood' not in col:
                            new_col='dlc_%s_%s' % col[1:]
                            raw[new_col]=dlc[col]
                    if save == True:
                        print('\t Saving updated Raw*.csv...')
                        raw.to_csv(path)
                        
                        meta_pnfn=path.parent.joinpath('metadata_%s.csv' % dataloc.path_to_rawfn(path)[4:])
                        print('\tSaving updated meta*.csv %s\n' % meta_pnfn)
                        meta.to_csv(meta_pnfn)
                    else:
                        return raw, meta
                else:
                    print('\t DLC content already added to Raw*.csv ... skipping.')
                
            else:
                print('\t %s does not exist in %s.\n' % ('dlc_analyze.h5',path.parent))

def unify_raw_to_csv(basepath,conds_inc=[],conds_exc=[],
                     force_replace=False, make_preproc=False, win=10):
    '''
    unify_raw_to_csv(basepath,conds_inc=[],conds_exc=[]):
            makes 
            a) Raw*.csv - file containing EthoVision raw data:
                - columns of samples for various measurements like mouse position (x,y)
            b) metadata.csv - dictionary with key experiment information (animal id, condition, etc).

            c) deeplabcut.csv - dataframe of body tracking if available

    Parameters
    ----------
    basepath : TYPE
        DESCRIPTION.
    conds_inc : TYPE, optional
        DESCRIPTION. The default is [].
    conds_exc : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    None.

    '''
    
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        xlsx_paths=dataloc.rawxlsx(basepath,inc,exc)
        if isinstance(xlsx_paths,Path):
            xlsx_paths=[xlsx_paths]
        for ii,path in enumerate(xlsx_paths):
            # First, let's check if there is already a .h5 file in this folder:    
            new_file_name=dataloc.path_to_rawfn(path) + '.csv'
            file_exists=os.path.exists(path.parent.joinpath(new_file_name))
            print('Inc[%d], file %d) %s generation...' % (i,ii,new_file_name))
            if (file_exists == False) or ((file_exists == True) and (force_replace == True)): 
                xlsxpath=path
                #If no .mat file exists, generate raw .csv from xlsx (slow)
                if ((file_exists == True) and (force_replace == True)):
                    print('\t .csv found but force_replace == True so file will be updated.')
                else:
                    print('\tNo .csv file found! Generating from .xlsx...')
                raw,meta=raw_meta_from_xlsx(xlsxpath)
                if isinstance(raw,pd.DataFrame):
                    #Set common boolean columns (but don't force it if these columns are not present):
                    common_bool=['im', 'm', 'laserOn','iz1','iz2']
                    for cb in common_bool:
                        if cb in raw.columns:
                            raw[cb]=raw[cb].astype('bool')
                    
                    #As of 1/21/21, expect raw to only have raw behavior and
                    # any changes will be in a separate preproc file made by
                    # behavior.preproc_raw()
                    
                    # #Assume a Raw*.csv now exists and add deeplabcut tracking to this .h5:
                    # dlcpath=dataloc.dlc_h5(path.parent) 
                    
                    raw, metadata = add_dlc_to_csv(path)
                    
                    if make_preproc == True:
                        raw_csv_to_preprocessed_csv(path.parent,
                                                    force_replace=force_replace,
                                                    win=win)
                    
                    #Write raw and meta to .csv files:
                    pnfn=path.parent.joinpath(new_file_name)
                    print('\tSaving %s\n' % pnfn)
                    raw.to_csv(pnfn)
                    
                    metadata=pd.DataFrame().from_dict(meta)
                    meta_pnfn=path.parent.joinpath('metadata_%s.csv' % dataloc.path_to_rawfn(path)[4:])
                    print('\tSaving %s\n' % meta_pnfn)
                    metadata.to_csv(meta_pnfn)
            else:
                # path_str=dataloc.path_to_description(path)
                print('\t %s already EXISTS in %s.\n' % (new_file_name,path.parent))
    print('Finished')   

def raw_csv_to_preprocessed_csv(basepath,conds_inc=[],conds_exc=[],force_replace=False,win=10):
    '''
    raw_csv_to_preprocessed_csv(basepath,conds_inc=[],conds_exc=[],force_replace=False,win=10):
            Takes
            a) Raw*.csv - file containing EthoVision raw data:
                - columns of samples for various measurements like mouse position (x,y)
            Returns: pre-processed:Preproc_*.csv file

            c) deeplabcut.csv - dataframe of body tracking if available

    Parameters
    ----------
    basepath : TYPE
        DESCRIPTION.
    conds_inc : TYPE, optional
        DESCRIPTION. The default is [].
    conds_exc : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    None.

    '''
    
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        csv_paths=dataloc.raw_csv(basepath,inc,exc)
        if isinstance(csv_paths,Path):
            xlsx_paths=[csv_paths]
        for ii,path in enumerate(csv_paths):
            # First, let's check if there is already a .csv file in this folder:    
            preproc_file_name=dataloc.path_to_preprocfn(path) + '.csv'
            file_exists=os.path.exists(path.parent.joinpath(preproc_file_name))
            print('Inc[%d], file %d) %s generation...' % (i,ii,preproc_file_name))
            if (file_exists == False) or ((file_exists == True) and (force_replace == True)): 
                rawpath=path
                #If no .mat file exists, generate raw .csv from xlsx (slow)
                if ((file_exists == True) and (force_replace == True)):
                    print('\t preproc*.csv found but force_replace == True so file will be updated.')
                else:
                    print('\tNo preproc*.csv file found- Generating from Raw*.csv...')
                raw,meta=csv_load(rawpath)
                if isinstance(raw,pd.DataFrame):
                    preproc,meta=behavior.preproc_raw(raw,meta,win=win)
                    
                    #Write raw and meta to .csv files:
                    pnfn=path.parent.joinpath(preproc_file_name)
                    print('\tSaving %s\n' % pnfn)
                    preproc.to_csv(pnfn)
                    
                    metadata=pd.DataFrame().from_dict(meta)
                    meta_pnfn=path.parent.joinpath('metadata_%s.csv' % dataloc.path_to_rawfn(path)[4:])
                    print('\tSaving %s\n' % meta_pnfn)
                    metadata.to_csv(meta_pnfn)
            else:
                # path_str=dataloc.path_to_description(path)
                print('\t %s already EXISTS in %s.\n' % (preproc_file_name,path.parent))
    print('Finished')   
    
def meta_sum_h5(basepath,conds_inc=[],conds_exc=[]):
    '''
        Generate a summary dataframe of metadata from .h5 files specified by
        inclusion criteria.
    '''
    df=pd.DataFrame
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        h5_paths=dataloc.rawh5(basepath,inc,exc)
        if isinstance(h5_paths,Path):
            h5_paths=[h5_paths]
        for ii,path in enumerate(h5_paths):
            par=h5_load_par(path)
            pardf=pd.DataFrame
            pardf=pardf.from_dict(par,orient='index').T #
            if ii==0:
                df=pardf
            else:
                df=df.append(pardf)
    cols_keep=['folder_anid',
               'stim_area',
               'cell_type',
               'opsin_type',
               'side',
               'protocol',
               'da_state',
               'etho_sex',
               'etho_arena',
               'etho_stim_info',
               'animal_id_mismatch',
               'possible_retrack',
               'experimenter',
               'exp_room_number',
               'etho_trial_number',
               'etho_trial_control_settings',]
    return df[cols_keep]

def meta_sum_csv(basepath,conds_inc=[],conds_exc=[]):
    '''
        Generate a summary dataframe of metadata from .csv files specified by
        inclusion criteria.
    '''
    df=pd.DataFrame
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        paths=dataloc.meta_csv(basepath,inc,exc)
        if isinstance(paths,Path):
            paths=[paths]
        if len(paths) > 0:
            for ii,path in enumerate(paths):
                pardf=meta_csv_load(path).loc[[0]]
                if ii==0:
                    df=pardf
                else:
                    df=pd.concat([df,pardf],axis=0)
        else:
            df=pd.DataFrame()
            print('Warning, no paths found.')
                
    cols_keep=['folder_anid',
               'stim_area',
               'cell_type',
               'opsin_type',
               'side',
               'protocol',
               'stim_n', 
               'stim_mean_dur',
               'da_state',
               'etho_sex',
               'etho_arena',
               'etho_stim_info',
               'animal_id_mismatch',
               'possible_retrack',
               'experimenter',
               'exp_room_number',
               'etho_trial_number',
               'etho_trial_control_settings',]
    df=df[cols_keep]
    df=df.reset_index().drop(['index'],axis=1)
    cols_rename=['anid','area','cell','opsin','side','proto','stim_n','stim_dur',
                 'da','sex','arena',
                 'stim','id_err','retrack','exper','room','trial','settings']
    rename_dict={cols_keep[i] : cols_rename[i] for i in range(len(cols_rename))} 
    
    return df.rename(rename_dict,axis=1)
    
def analyze_df(fun,basepath,conds_inc=[],conds_exc=[]):
    
    all_out=dict()
    for i,inc in enumerate(conds_inc):
        exc=conds_exc[i]
        h5_paths=dataloc.rawh5(basepath,inc,exc)
        outs=[]
        field=str(i)
        if isinstance(h5_paths,Path):
            h5_paths=[h5_paths]
        for ii,path in enumerate(h5_paths):
            data,par=h5_load(path)
            out=fun(data,par)
            outs.append(out)
        all_out[field]=outs
    return all_out

def h5_store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    #store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5_load(filename):
    store = pd.HDFStore(filename)
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    store.close()
    return data, metadata

def csv_load(exp_path,columns='All',method='raw'):
    if method == 'raw':
        rawfn=dataloc.path_to_rawfn(exp_path) + '.csv'
    else:
        rawfn=dataloc.path_to_preprocfn(exp_path) + '.csv'
    pnfn=exp_path.parent.joinpath(rawfn)
    print('Loading %s\n' % pnfn)
    a=pd.read_csv(pnfn,nrows=1)
    if isinstance(columns,list):
        use_cols=columns
        raw=pd.read_csv(pnfn,usecols=use_cols)
    elif columns=='Minimal':
        use_cols=['time', 'x', 'y','dir', 'vel', 'im',
       'laserOn'] 
        if method == 'raw':
            if 'dlc_is_rearing_logical' in a.columns:
                use_cols=use_cols + ['dlc_is_rearing_logical']
        else:
            if 'rear' in a.columns:
                use_cols=use_cols + ['rear']

        raw=pd.read_csv(pnfn,usecols=use_cols)
    else:
        raw=pd.read_csv(pnfn)
        
    
    meta_pnfn=exp_path.parent.joinpath('metadata_%s.csv' % dataloc.path_to_rawfn(exp_path)[4:])
    metadata=meta_csv_load(meta_pnfn)
    return raw, metadata

def meta_csv_load(filename):
    metadata=pd.read_csv(filename,index_col='Unnamed: 0')
    return metadata

def h5_load_par(filename):
    store = pd.HDFStore(filename)
    metadata = store.get_storer('mydata').attrs.metadata
    store.close()
    return metadata

def meta_from_mat(pn):
    '''
    Extract parameters from existing .mat file via scipy.io.loadmat output (dict of arrays)

    Parameters
    ----------
    pn : path a matlab .mat file  
        
    Returns
    meta : dict containing experiment information

    '''
    
    mat=mat_file.load(pn,verbose=False)
    anid=dataloc.path_to_animal_id(str(pn))
    
    #Organize parameters:
    if 'fs' in mat:
        fs=mat['fs'][0][0]
    else:
        fs=1/np.mean(np.diff(mat['time'][0:].flatten())) #usually ~29.97 fps
    possible_retrack=0
        
    pars=mat_file.dtype_array_to_dict(mat,'meta')
    nold = mat_file.dtype_array_to_dict(mat,'noldus')
    
    #Check if there is any ambiguity about which mouse is in the file:
    if isinstance(nold['Mouse_ID'],str):
        if (nold['Mouse_ID'] != pars['anID']) \
            or (nold['Mouse_ID'] != anid):
            anid_mismatch=1
        else:
            anid_mismatch=0
    else:
        anid_mismatch=np.nan
        
    #Examine tracking source to determine if video is retracked or which room is was likely filmed in:
    if isinstance(nold['Tracking_source'],str):
        track_source=nold['Tracking_source']
        if 'Basler GenICam' in track_source:
            room_num=216
        elif 'Euresys PICOLO' in track_source:
            room_num=228
        else:
            room_num=np.nan
    else:
        room_num=np.nan
        possible_retrack=1
        track_source=np.nan
    
    # Look for light information if available:
    light_method='Stim_'
    if 'LED_Dial' in nold:
        light_method='LED_Dial'
    elif 'Dial' in nold:
        light_method='Dial'
    
    # Look for mouse sex information if available:
    if 'Sex' in nold:
        sex=nold['Sex']
    elif 'Gender' in nold:
        sex=nold['Gender'] # sex='Gender' #Older version
    else:
        sex=np.nan
        
    # Look for mouse depletion status if available:
    if 'Days_after_Depletion' in nold:
        dep_status=nold['Days_after_Depletion']
    elif 'Days_after' in nold:
        dep_status = nold['Days_after']
    elif 'Depletion_Status' in nold:
        dep_status = nold['Depletion_Status']
    
    if 'Test' in nold:
        etho_exp=nold['Test']
    else:
        etho_exp=np.nan
        
    meta={
        'etho_animal_id': nold['Mouse_ID'],
        'etho_da_state':dep_status,
        'etho_sex': sex,
        'etho_treatment': nold['Treatment'],
        'etho_video_file': nold['Video_file'],
        'etho_exp_date':nold['Start_time'],
        'etho_rec_dur':nold['Trial_duration'],
        'etho_trial_control_settings':nold['Trial_Control_settings'],
        'etho_trial_number': nold['Trial_name'].rsplit()[1],#'etho_experiment_file': mat['noldus'][0][0][0][0],
        'etho_arena':nold['Arena_settings'], #Two zone arena
        'etho_stim_info' :nold[light_method], #Green = Dial 768, etc.
        'etho_exp_type' : etho_exp, # 10x30 etc.   
        'etho_tracking_source' : track_source,
        'exp_room_number': room_num,
        'experimenter':pars['exp'].split('_')[-1][0:2],
        'anid':pars['anID'],
        'folder_date':pars['exp'].split('_')[-1][2:],
        'folder_anid':anid,
        'protocol': pars['protocol'],
        'side': pars['side'],
        'opsin_type':pars['opsin_type'],
        'cell_type':pars['cell_type'],
        'da_state':pars['da_state'],
        'stim_area':pars['stim_area'],
        'animal_id_mismatch':anid_mismatch,
        'possible_retrack':possible_retrack,
        'fs':fs, #Default frames / second but should confirm in each video if possible        
        }
    if 'Experiment' in nold:
        meta['etho_experiment_file']=nold['Experiment']
    else:
        meta['etho_experiment_file']=np.nan
        
    inc=['task','exp_start','exp_end','task_start','task_stop','zone']
    for f in inc:
        if f in mat:
            if mat[f].size > 0:
                meta[f]=mat[f].flatten()[0]
            else:
                meta[f]=[]
                
    #Incorporate stimulus info into the meta dictionary:
    use_keys=['laserOnTimes','laserOffTimes']
    rename=['stim_on','stim_off']
    meta['stim_amp_mw']=1 #By default, 1 mW
    proto=str(pn).split(sep)[-3]
    meta['stim_proto']=proto
    if ('mw' in proto) or ('mW' in proto):
        meta['stim_amp_mw']=int(proto.split('_')[-1].split('m')[0])
    for i,key in enumerate(use_keys):
        meta[rename[i]]=mat[key].flatten()
    stim_dur=[]
    for i,onset in enumerate(meta['stim_on']):
        if i< len(meta['stim_off']):
            stim_dur.append(meta['stim_off'][i]-onset)
        else: #If recording ended before stimulus shut off for some reason
            stim_dur.append(np.nan)
    meta['stim_dur']=np.array(stim_dur)
    
    meta['stim_n']=np.nansum(not np.isnan(meta['stim_on']))
    meta['stim_mean_dur']=meta['stim_dur'].mean()
    return meta



def raw_from_mat(pn):
    '''
    raw_df_from_mat(pn) - Import raw ethovision data as columns of a pandas dataframe.
    Parameters
    ----------
    pn : Path (from pathlib import Path)
        File path to a .mat file, e.g. 
       '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPi/Naive/CAG/Arch/Bilateral/10x30/AG6151_3_CS090720/Raw_AG6151_3_CS090720.mat'

    Returns
    -------
    df : dataframe
        Dataframe containing columns of raw data exported from ethovision.

    '''
    
    mat=mat_file.load(pn)
    
    #Include only keys that are arrays of frame sample values:
    use_keys=['time', 'x', 'y', 'x_nose', 'y_nose', 'x_tail', 'y_tail', 'area',
       'delta_area', 'elon', 'dir', 'dist', 'vel', 'im', 'm', 'quarter_rot_cw',
       'quarter_rot_ccw', 'iz1', 'iz2', 'laserOn', 'ambulation', 'fine_move',
       'net_cw_rots']
        
    #Create new dict:
    raw_dat={}
    for key in use_keys:
        if key in mat:
            raw_dat[key]=mat[key].flatten()
        else:
            print('Warning: %s key not found, nan-filling' % key)
            raw_dat[key]=np.full_like(mat['time'].flatten(),np.nan)
    
    #Create a dataframe from dict:
    df=pd.DataFrame.from_dict(raw_dat)
    return df

def raw_meta_from_xlsx(pn):
    # Next, we will read in the data using the imported function, read_excel()
    path_str=dataloc.path_to_description(pn)
    print('\tLoading ~%s_Raw.xlsx...' % path_str)
    df=pd.read_excel(pn,sheet_name=None,na_values='-',header=None) #Key addition
    sheets=[sheet for sheet in df.keys()]
    if len(sheets) > 1:
        raw=raw_from_xlsx(df)
        meta=meta_from_xlsx(df,pn)
        stim=stim_from_xlsx(df,pn)
        meta['fs']=1/np.mean(np.diff(raw['time'][0:]))
        for key,value in stim.items():
            meta['stim_' + key]=value
        
        #Also add in: exp_end, task_start, task_stop
        meta=check_meta(df,raw,meta)        
    else:
        print('\tWARNING: This .xlsx file does not contain sufficient number of sheets for further processing.')
        print(pn)
        raw=[]
        meta=[]
    return raw,meta

def get_header_from_rawdf(df,sheet=0):
    '''
        Take in dictionary of dataframes made from Raw*.xlsx ethovision file 
        and return header.
            
        INPUT: dictionary of 3 panadas.DataFrame sheets imported from Raw*.xlsx
        OUTPUT: pandas.DataFrame, 1-row, columns w/ names of header entries
    '''
    
    sheets=np.array([key for key in df.keys()]) #First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    if isinstance(sheet,int):
        use_sheet=sheets[sheet]
    else:
        use_sheet=sheets[sheet][0]
        
    header_amount=int(df[use_sheet][1][0])
    temp=df[use_sheet]
    header=temp.iloc[0:(header_amount-3),0:2]
    header= header.set_index(0).transpose()
    return header

def get_header_and_sheet_rawdf(df,sheet=0):
    '''
        Take in dictionary of dataframes made from Raw*.xlsx ethovision file and
        desired sheet number, return header and sheet as clean dataframes.
            
        INPUTs:
            df - dictionary of 3 panadas.DataFrame sheets imported from Raw*.xlsx
            sheet - sheet number to use, by default sheet = 0
        OUTPUTs:
            header -pandas.DataFrame, 1-row, columns w/ names of header entries
            data - pandas.DataFrame of requested sheet from df
    '''
    sheets=np.array([key for key in df.keys()])#First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    header=get_header_from_rawdf(df,sheet)
    header_amount=int(header['Number of header lines:'])
    if isinstance(sheet,int):
        use_sheet=sheets[sheet]
    else:
        use_sheet=sheets[sheet][0]
        
    temp=df[use_sheet]
    
    temp=temp.drop([i for i in range(0,header_amount-2)])
    temp=temp.reset_index(drop=True)
    temp.columns = temp.iloc[0]
    data = temp.drop(0) 
  
    #Check if first row has strings showing the units instead of data:
    first_val=data.iloc[0,0]
  
    if isinstance(first_val, str):
        # print('\tDetected unnecessary row 0, dropping..')
        data=data.drop(1).reset_index(drop=True)
    #Remove any columns with nan labels
    data = data.loc[:, data.columns.notnull()]
    return header, data

def check_meta(df,raw,meta):
    '''
        Take existing meta and use raw .xlsx (in df) to verify / clarify correct
        task type and start/stop times.
    '''
    keys=['Trial' in key for key in df.keys()]
    _,data=get_header_and_sheet_rawdf(df,keys)
    data.iloc[data.isna()]='-' # Return to '-' in this case
    
    
    # Attempt to check the Raw*.xlsx file
    rule_time = data['Trial time'].values
    rule_order = [3, 5, 4, 6] 
    rule = [] #List of columns
    for j,rn in enumerate(rule_order):
        temp=data.iloc[:,rn]
        temp[pd.isna(temp)] = '-'
        rule.append(temp)
        
   
    # 1) Unify experiment descriptions to "avoidance", "trigger", "10x30" 
    # 2) Determine if there's any reason to think the file is in the wrong folder (i.e. task mis-classified)
    iszone=[('zone_' in a) or ('Zone' in a) for a in rule[2]]
    if any(iszone):
        meta['task'] = 'avoidance' 
    elif 'right_trig' in raw.columns:
        meta['task'] = 'trigger'
    elif ('rescue' in meta['protocol']) or ('10x' in meta['protocol']):
        meta['task'] = '10x_stim'
    else:
        meta['task'] = meta['protocol']
    
    meta['active_trigger']=''
    if meta['task'] == 'trigger':
        trig=meta['protocol'].split('_')[1]
        if trig == 'l':
            meta['active_trigger']='Left'
        elif trig=='r':
            meta['active_trigger']='Right'

    # Clarify best way to describe experiment vs. task onset (i.e. is there a 
    # 'pre-' / 'baseline' period?)
    meta['exp_start'] = raw['time'].values[0]    
    meta['exp_end']=raw['time'].values[-1]

    event_ends =np.array(['becomes inactive' in i for i in data['Event']])
    event_starts =np.array(['becomes active' in i for i in data['Event']])
    baseline = np.array([any([x in y for x in ['Baseline','base','Habituation']]) for y in rule[2]])
    if any (event_ends & baseline):
        meta['task_start'] = rule_time[event_ends & baseline][0]
    else:
        print('\tNo baseline period found, available events:')
        for val in pd.unique(rule[2]):
            print('\t' + val)
        meta['task_start'] = ''
        print('\n')
    if meta['task_start'] == meta['exp_end']:
        #Entire trial likely 'free-running' and treated as baseline:
        meta['task_start']=''
        
    task_stop=np.array([any([x in y for x in ['Time (1)','post_period']]) for y in rule[2]])
    if any(event_starts & task_stop):
        meta['task_stop'] = rule_time[task_stop & event_starts][0]
    else:
        print('\tNo task stop time found, fields available:')
        for val in pd.unique(rule[2]):
            print('\t' + val)
        meta['task_stop'] = ''
        print('\n')
        
    #Check if zone or other task has blink state.
    #(This is often used if mouse stays in stimulated zone > certain time)
    meta['has_blink_state']=False
    blink_present=np.array([any([x in y for x in ['Blink On','blink']]) for y in rule[2]])
    if any(blink_present):
        meta['has_blink_state'] = True
    #
    
    if meta['task_stop'] == '':
        #Trial recording might have ended prematurely, estimate task_stop:
        if meta['task'] == 'trigger':
            #Task ends when last stim turns off, or 20 minutes after start, whichever is possible
            # stop=meta['stim_off'][-1]
            # if np.isnan(stop):
            print('\tNo task_stop found, estimating 20min after task_start.')
            stop=meta['task_start'] + (20 * 60) #20 min in seconds
            meta['task_stop'] = stop
        elif meta['task'] == '10x30':
            print('\tNo task_stop found, estimating 30min after task_start.')
            meta['task_stop']=meta['task_start'] + (30 * 60) #30 min in seconds
        elif meta['task'] == 'aversion':
            print('\tNo task_stop found, estimating 30min after task_start.')
            meta['task_stop']=meta['task_start'] + (30 * 60) #30 min in seconds
        elif meta['task'] == '30min':
            meta['task_stop'] = 30 * 60 #30 min in seconds
        elif any(i in meta['task'] for i in ['60min','60min_psem','60min_saline']):
            meta['task_stop'] = 60 * 60 #60 min in seconds
        else:
            meta['task_stop']=meta['exp_end']
    meta['no_trial_structure']=False   
    if meta['task_start'] == '':
        if meta['stim_n']>0:
            #If there is evidence of stim but task start still empty, set task start to 30s before first stim
            meta['task_start']=meta['stim_on'][0]
            print('\tNo trial start detected but stim detected, task start set to time of 1st stim: %4.1fs' % meta['task_start'])
        else:
            #Assume this is a free-running recording without structure and task starts at 0:
            meta['task_start']=raw['time'].values[0]
            meta['no_trial_structure']=True
            meta['task_stop']=raw['time'].values[-1]
            print('\tFree running trial detected. Setting task start time to time 0.')
    
    if len(meta['stim_on']) > len(meta['stim_off']):
        meta['stim_off']=np.hstack((meta['stim_off'],meta['exp_end']))
        
    if len(meta['stim_off']) > len(meta['stim_on']):
        meta['stim_off']=meta['stim_off'][0:len(meta['stim_on'])]
    
    if any(meta['stim_dur'] < 0):
        # stim_dur_negative_detected
        print('\tProblem with stim time alignment as negative stim durations detected.')
        # stim_dur_negative_detected
    meta['task_prestart_warning']=False
    if meta['stim_on'][0] < meta['task_start']:
        meta['task_prestart_warning']=True
        meta['task_start'] = meta['stim_on'][0]
        print('\tWarning, first stim occurred before "task_start" time! Setting task start to %4.1fs' % meta['task_start'])
    meta['task_overrun_warning']=False
    if (meta['task'] == 'aversion') and (meta['task_stop'] > (41*60)): # All of these should be titrated appropriately
        meta['task_overrun_warning']=True
    elif (meta['task'] == 'trigger') and (meta['task_stop'] > (36*60)):
        meta['task_overrun_warning']=True
    elif (meta['task'] == '10x30') and (meta['task_stop'] > (46 * 60)):
        meta['task_overrun_warning']=True
    elif (meta['task_stop'] == meta['exp_end']) and (not meta['no_trial_structure']):
        meta['task_overrun_warning']=True
    
    if meta['task_overrun_warning']:
        print('\tTask stop time appears too long, trial flagged as overruning.')
    
    return meta
    
def stim_from_xlsx(df,pn):
    keys=['Hardware' in key for key in df.keys()]
    
    if any(keys):
        header,data=get_header_and_sheet_rawdf(df,keys)  
        command=data['Command/Signal']=='signal'
        trial_start = data['Trial time'] > 0
        name = data['Name']=='Is output 1 High'
        val_on = data['Value'] == 1
        val_off = data['Value'] == 0
        dev= [[x in ['Custom Hardware 1','Laser','Laser control']] for x in data['Device']]
        stim_on = command & name & val_on & dev & trial_start
        stim_off = command & name & val_off & dev & trial_start
        stim={'on': data['Recording time'][stim_on].values.astype('float'),
              'off': data['Recording time'][stim_off].values.astype('float'),
              }
    else:
        stim={'on':'','off':''}
        

    proto=str(pn).split(sep)[-3]
    stim['proto']=proto
    if len(stim['on']) == 0:
        if '10x30' in proto: #Stim times occur at well-known times, can be inferred in few cases of missing values
            stim['on']=[x for x in range(630,2550,(60*3+30))] #infer
            stim['off']=[i + 30 for i in stim['on']]
        else:
            stim['on']=[np.nan]
            stim['off']=[np.nan]
    stim_dur=[]
    for i,onset in enumerate(stim['on']):
        if i< len(stim['off']):
            stim_dur.append(stim['off'][i]-onset)
        else: #If recording ended before stimulus shut off for some reason
            stim_dur.append(np.nan)
    # debug_error
    stim['dur']=np.array(stim_dur)
    stim['amp_mw']=1 #By default, 1 mW
    if not any(np.isnan(stim['on'])):
        stim['n']=len(stim['on'])
    else:
        stim['n']=0
    if ('mw' in proto) or ('mW' in proto):
        temp_amp=proto.split('_')[1].split('m')[0]
        if 'p' in temp_amp:
            use_amp = int(temp_amp.split('p')[1])/10
        else:
            use_amp = int(temp_amp)
        stim['amp_mw']=use_amp
    if len(stim['dur']) > 0:        
        stim['mean_dur']=np.nanmean(stim['dur'])
    else:
        stim['mean_dur']=0
        # stim['dur']=[np.nan]
    return stim

def meta_from_xlsx(df,pn):
    '''
    Read header information from EthoVision raw .xlsx dataframe loaded via pandas.read_excel

    Parameters
    ----------
    Input: 
    -------
    pn: Path
        File path to raw .xlsx file using Path() 

    Returns
    -------
    meta: Dict
        Dictionary containing all relevant information about the given experiment,
        gleaned from header information in raw data df

    '''

    nold=get_header_from_rawdf(df) #Header info, largely added by experimenter during trial
    header_amount=int(nold['Number of header lines:'])

    str_pn=str(pn)
    anid=dataloc.path_to_animal_id(str_pn)
    
    #Check if there is any ambiguity about which mouse is in the file:
    if 'Animal ID' in nold.columns:
        nold_an=nold['Animal ID'].values[0]
    elif 'Mouse ID' in nold.columns:
        nold_an=nold['Mouse ID'].values[0]
    elif 'Subject ID' in nold.columns:
        nold_an=nold['Subject ID'].values[0]
    else:
        nold_an=np.nan
    if isinstance(nold_an,str):
        if (nold_an != anid):
            anid_mismatch=1
        else:
            anid_mismatch=0
    else:
        anid_mismatch=np.nan
        
    #Examine tracking source to determine if video is retracked or which room is was likely filmed in:
    possible_retrack=0
    if isinstance(nold['Tracking source'].values[0],str):
        track_source=nold['Tracking source'].values[0]
        if 'Basler GenICam' in track_source:
            room_num=216
        elif 'Euresys PICOLO' in track_source:
            room_num=228
        else:
            room_num=np.nan
            possible_retrack=1
    else:
        room_num=np.nan
        possible_retrack=1
    
    # Look for light information if available:
    light_method=''
    if 'LED Dial' in nold.columns:
        light_method='LED Dial'
    elif 'Dial' in nold.columns:
        light_method='Dial'
    elif 'Stim #' in nold.columns:
        light_method='Stim #'
    
    # Look for mouse sex information if available:
    if 'Sex' in nold.columns:
        sex=nold['Sex'].values[0]
    elif 'Gender' in nold.columns:
        sex=nold['Gender'].values[0] # sex='Gender' #Older version
    else:
        sex=np.nan
        
    # Look for mouse depletion status if available:
    if 'Days after Depletion' in nold:
        dep_status=nold['Days after Depletion'].values[0]
    elif 'Days after' in nold:
        dep_status = nold['Days after'].values[0]
    elif 'Depletion Status' in nold:
        dep_status = nold['Depletion Status'].values[0]
    elif '# Days after Depletion' in nold:
        dep_status = nold['# Days after Depletion'].values[0]
    else:
        dep_status='N/A'
        
    stim_details=''
    if light_method != '':
        stim_details=nold[light_method].values[0]
        
    if 'Test' in nold:
        etho_exp=nold['Test'].values[0]
    else:
        etho_exp=np.nan
    meta={
        'anid':anid,
        'protocol':str_pn.split(sep)[-3], 
        'side': str_pn.split(sep)[-4],
        'opsin_type':str_pn.split(sep)[-5],
        'cell_type':str_pn.split(sep)[-6],
        'da_state':str_pn.split(sep)[-7],
        'stim_area':str_pn.split(sep)[-8],
        'task': str_pn.split(sep)[-3], #will be changed in check_meta()
        'experimenter':str_pn.split(sep)[-2].split('_')[-1][0:2],
        'exp_room_number':room_num,
        'fs':29.97, # placeholder default frames / second but should confirm in each video if possible
        'exp_start': 0, #will be updated in check_meta()
        'exp_end':[], #will be updated in check_meta()
        'task_start':[], #will be updated in check_meta()
        'task_stop':[], #will be updated in check_meta()
        'etho_animal_id': nold_an,
        'etho_da_state': dep_status,
        'etho_sex':sex,
        'etho_video_file':nold['Video file'].values[0],
        'etho_exp_date':nold['Video start time'].values[0],
        'etho_rec_dur':nold['Recording duration'].values[0],
        'etho_trial_control_settings':nold['Trial Control settings'].values[0],
        'etho_trial_number': int(nold['Trial name'].values[0].rsplit()[1]),
        'etho_experiment_file': nold['Experiment'].values[0],
        'etho_arena': nold['Arena settings'].values[0], #Two zone arena
        'etho_stim_info' : stim_details, #Green = Dial 768, etc.
        'etho_exp_type' : etho_exp, # 10x30 etc.   
        'etho_tracking_source' : nold['Tracking source'].values[0],
        'folder_anid':anid,
        'folder_date':str_pn.split(sep)[-2].split('_')[-1][2:],
        'animal_id_mismatch':anid_mismatch,
        'possible_retrack':possible_retrack,
        }
    if 'zone' in meta['protocol']:
        meta['zone']='%s %s' % (meta['protocol'].split('_')[0].capitalize(),
                        meta['protocol'].split('_')[1])

    
    return meta


def raw_from_xlsx(df):
    """ raw_from_xlsx(pn) takes in a path to a Raw ethovision export excel file,
        
      0) load the .xlsx raw data file and generate .h5 from scratch
      1) imports the first sheet of the excel file as a dataframe using pandas
      2) checks to see if first row is units, if so: drops that row
      3) Exports the data to raw object in .h5 file in the same directory.
      Future things to add: 

      path='/content/drive/My Drive/Colab Notebooks/Gittis Lab Python Code/example_data/ethovision_data/Raw data-bi_two_zone_rm216_v1-Trial 41.xlsx'
      >>> newpath= raw_excel2h5(path)
      >>> newpath = '/content/drive/My Drive/Colab Notebooks/Gittis Lab Python Code/example_data/ethovision_data/Raw_AG5362_3_BI022620.h5'
      
    """
        
    header,data=get_header_and_sheet_rawdf(df,0)
    data=rename_xlsx_columns(data)
    drop_cols=['rec_time','m_cont']
    for dc in drop_cols:
        if dc in data.columns:
            data.drop(columns=dc,inplace=True)
    data.reset_index(inplace=True,drop=True)
    
    # Either way, return the path/filename of the .csv file:
    return data



def rename_xlsx_columns(df):
    '''
        Input: 'df' = raw .xlsx dataframe loaded via pandas.read_excel
        
        Output: 'df' = raw .xlsx dataframe with simplified / clarified column names
    '''
    rename_dict = {  'Trial time':'time', 'Trial time (s)':'time',
                     'Recording time':'rec_time', 'Recording time (s)':'rec_time',
                     'X center': 'x', 'X center (cm)': 'x',
                     'Y center':'y', 'Y center (cm)':'y',
                     'X nose':'x_nose', 'X nose (cm)':'x_nose',
                     'Y nose':'y_nose', 'Y nose (cm)':'y_nose',
                     'X tail':'x_tail', 'X tail (cm)':'x_tail',
                     'Y tail':'y_tail', 'Y tail (cm)':'y_tail',
                     'Area':'area', 'Area (cm²)': 'area',
                     'Areachange': 'delta_area', 'Areachange (cm²)': 'delta_area',
                     'Elongation':'elon',
                     'Direction':'dir','Direction (deg)' : 'dir',
                     'Distance moved':'dist', 'Distance moved (cm)':'dist',
                     'Velocity':'vel', 'Velocity (cm/s)':'vel',
                     'Mobility state(Immobile)':'im','Mobility state(Immobile)':'im','Movement(Not Moving / Center-point)':'im',
                     'Mobility state(Mobile)':'m','Movement(Moving / Center-point)':'m',
                     'Mobility continuous':'m_cont','Mobility':'m_cont',
                     'Rotation':'quarter_rot_cw',
                     'Rotation 2':'quarter_rot_ccw',
                     'Full CW Rotation':'full_rot_cw',
                     'Full CCW Rotation':'full_rot_ccw',
                     'In zone(Zone 1 / Center-point)':'iz1','In zone(Zone 1 / center-point)':'iz1',
                     'In zone(Zone 2 / Center-point)': 'iz2','In zone(Zone 2 / center-point)':'iz2',
                     'In zone(dot_side / Center-point)':'dot_side',
                     'In zone(stripe_side / Center-point)':'stripe_side',
                     'Hardware state':'laserOn',
                     'Left trigger':'left_trig',
                     'Right trigger': 'right_trig'
                     }
    if 'Result 1' in df.columns:
        df=df.drop('Result 1',axis=1)
    return df.rename(rename_dict,axis=1)


    
    
    # keep.fine_move= ~keep.ambulation & ~keep.im; %Note: keep.im is the ethovision 'immobile' output