import pandas as pd
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
            a) params - dictionary with key experiment information (animal id, condition, etc).
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
                    raw=raw_from_mat(matpath) #Will add 'raw', 'params', and 'stim' to .h5
                    params=params_from_mat(matpath)
                else: #If no .mat file exists, generate raw .h5 from xlsx (slow)
                    print('\tNo .mat file found! Generating from .xlsx...')
                    raw,params=raw_params_from_xlsx(xlsxpath)

                
                #Set common boolean columns (but don't force it if these columns are not present):
                common_bool=['im', 'm', 'laserOn','iz1','iz2']
                for cb in common_bool:
                    if cb in raw.columns:
                        raw[cb]=raw[cb].astype('bool')
                
                #Improved velocity measure:
                
                raw['vel']=behavior.smooth_vel(raw,params,win)
                params['vel_smooth_win_ms']=win/params['fs'] * 1000 # ~333ms
                
                thresh=2; #cm/s; Kravitz & Kreitzer 2010
                dur=0.5; #s; Kravitz & Kreitzer 2010
                raw=add_amb_to_raw(raw,params,thresh,dur,im_thresh=1,im_dur=0.25) #Thresh and dur
                params['amb_vel_thresh']=thresh
                params['amb_dur_criterion_ms']=dur
                
                #Assume a Raw*.h5 now exists and add deeplabcut tracking to this .h5:
                dlcpath=dataloc.dlc_h5(path.parent)
                
                #add_deeplabcut(path) # still need to write this function -> integrated into raw? something else?
                
                #Write raw and params to .h5 file:
                pnfn=path.parent.joinpath(new_file_name)
                print('Saving %s\n' % pnfn)
                h5_store(pnfn,raw,**params)
                
            else:
                # path_str=dataloc.path_to_description(path)
                print('\t %s already EXISTS in %s.\n' % (new_file_name,path.parent))
    print('Finished')
    
def unify_to_csv(basepath,conds_inc=[],conds_exc=[],force_replace=False,win=10):
    '''
    unify_to_csv(basepath,conds_inc=[],conds_exc=[]):
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
                print('\tNo .csv file found! Generating from .xlsx...')
                raw,params=raw_params_from_xlsx(xlsxpath)
                
                #Set common boolean columns (but don't force it if these columns are not present):
                common_bool=['im', 'm', 'laserOn','iz1','iz2']
                for cb in common_bool:
                    if cb in raw.columns:
                        raw[cb]=raw[cb].astype('bool')
                
                #Improved velocity measure:
                raw['vel']=behavior.smooth_vel(raw,params,win)
                params['vel_smooth_win_ms']=win/params['fs'] * 1000 # ~333ms
                
                thresh=2; #cm/s; Kravitz & Kreitzer 2010
                dur=0.5; #s; Kravitz & Kreitzer 2010
                raw=add_amb_to_raw(raw,params,thresh,dur,im_thresh=1,im_dur=0.25) #Thresh and dur
                params['amb_vel_thresh']=thresh
                params['amb_dur_criterion_ms']=dur
                
                #Assume a Raw*.csv now exists and add deeplabcut tracking to this .h5:
                dlcpath=dataloc.dlc_h5(path.parent) 
                
                #add_deeplabcut(path) # still need to write this function -> integrated into raw? something else?
                
                #Write raw and params to .csv files:
                pnfn=path.parent.joinpath(new_file_name)
                print('Saving %s\n' % pnfn)
                raw.to_csv(pnfn)
                
                metadata=pd.DataFrame().from_dict(params)
                # for key in params.keys():
                #     metadata.loc[key,0]=params[key]
                meta_pnfn=path.parent.joinpath('metadata_%s.csv' % dataloc.path_to_rawfn(path)[4:])
                print('Saving %s\n' % meta_pnfn)
                metadata.to_csv(meta_pnfn)
            else:
                # path_str=dataloc.path_to_description(path)
                print('\t %s already EXISTS in %s.\n' % (new_file_name,path.parent))
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
    cols_rename=['anid','area','cell','opsin','side','proto','da','sex','arena',
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

def csv_load(exp_path):
    rawfn=dataloc.path_to_rawfn(exp_path) + '.csv'
    pnfn=exp_path.parent.joinpath(rawfn)
    print('Loading %s\n' % pnfn)
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

def params_from_mat(pn):
    '''
    Extract parameters from existing .mat file via scipy.io.loadmat output (dict of arrays)

    Parameters
    ----------
    pn : path a matlab .mat file  
        
    Returns
    params : dict containing experiment information

    '''
    
    mat=mat_file.load(pn,verbose=False)
    anid=dataloc.path_to_animal_id(str(pn))
    
    #Organize parameters:
    if 'fs' in mat:
        fs=mat['fs'][0][0]
    else:
        fs=1/np.mean(np.diff(mat['time'][0:].flatten())) #usually ~29.97 fps
    possible_retrack=0
        
    pars=mat_file.dtype_array_to_dict(mat,'params')
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
        
    params={
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
        params['etho_experiment_file']=nold['Experiment']
    else:
        params['etho_experiment_file']=np.nan
        
    inc=['task','exp_start','exp_end','task_start','task_stop','zone']
    for f in inc:
        if f in mat:
            if mat[f].size > 0:
                params[f]=mat[f].flatten()[0]
            else:
                params[f]=[]
                
    #Incorporate stimulus info into the params dictionary:
    use_keys=['laserOnTimes','laserOffTimes']
    rename=['stim_on','stim_off']
    params['stim_amp_mw']=1 #By default, 1 mW
    proto=str(pn).split(sep)[-3]
    params['stim_proto']=proto
    if ('mw' in proto) or ('mW' in proto):
        params['stim_amp_mw']=int(proto.split('_')[-1].split('m')[0])
    for i,key in enumerate(use_keys):
        params[rename[i]]=mat[key].flatten()
    stim_dur=[]
    for i,onset in enumerate(params['stim_on']):
        if i< len(params['stim_off']):
            stim_dur.append(params['stim_off'][i]-onset)
        else: #If recording ended before stimulus shut off for some reason
            stim_dur.append(np.nan)
    params['stim_dur']=np.array(stim_dur)
    
    params['stim_n']=len(params['stim_on'])
    params['stim_mean_dur']=params['stim_dur'].mean()
    return params

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

def raw_params_from_xlsx(pn):
    
    # Next, we will read in the data using the imported function, read_excel()
    path_str=dataloc.path_to_description(pn)
    print('\tLoading ~%s_Raw.xlsx...' % path_str)
    df=pd.read_excel(pn,sheet_name=None,na_values='-',header=None) #Key addition
    raw=raw_from_xlsx(df)
    params=params_from_xlsx(df,pn)
    stim=stim_from_xlsx(df,pn)
    params['fs']=1/np.mean(np.diff(raw['time'][0:]))
    for key,value in stim.items():
        params['stim_' + key]=value
    
    #Also add in: exp_end, task_start, task_stop
   

    return raw,params

def get_header_from_rawdf(df):
    '''
        Take in dictionary of dataframes made from Raw*.xlsx ethovision file 
        and return header.
            
        INPUT: dictionary of 3 panadas.DataFrame sheets imported from Raw*.xlsx
        OUTPUT: pandas.DataFrame, 1-row, columns w/ names of header entries
    '''
    sheets=[key for key in df.keys()] #First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    header_amount=int(df[sheets[0]][1][0])
    temp=df[sheets[0]]
    header=temp.iloc[0:(header_amount-4),0:2]
    header= header.set_index(0).transpose()
    return header

def trialinfo_from_xlsx(df,params):
    sheets=[key for key in df.keys()] #First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    header=get_header_from_rawdf(df)
    header_amount=int(header['Number of header lines:'])
    temp=df[sheets[2]]
    temp=temp.drop([i for i in range(0,header_amount-3)])
    temp=temp.rename({col:temp[col][header_amount-3] for col in temp.columns},axis='columns')
    data=temp.drop(header_amount-3)
    
    rule_time=data['Trial time'].values
    rule_order=[3, 5, 4, 6]
    rule=[]
    for j,rn in enumerate(rule_order):
        temp=data.iloc[:,rn]
        temp[pd.isna(temp)]='-'
        rule.append(temp)
    iszone=[('zone' in a) or ('Zone' in a) for a in rule[2]]
    if any(iszone) and ('stripe' not in data.columns):
        params['task']='avoidance'
        zone_name=rule[2][iszone].iloc[0][0:6]
        if zone_name == 'zone_s':
            # params['zone']=
    if len(params['stim_on'])>0:
        params['task_start']=params['stim_on'][0]-30
    else:
        params['task_start']=0
    params['exp_end']=raw['time'].values[-1]
    if len(params['stim_on']) > len(params['stim_off']):
        params['stim_off']=np.hstack((params['stim_off'],params['exp_end']))
    params['task_stop']='multi' # see convert_raw_xlsx.m for notes
#   keep.task_stop=rule_time(find(contains(rule{2},{'Time (1)','post_period'}),1,'first'));
#                     if isempty(keep.task_stop)
    
    params['task_stop']=
    if params['task'] == 'trigger':
        if len(params['stim_off']) > 0:
                params(['task_stop'])=params['stim_off'][-1]
#                         if strcmp(keep.task,'trigger')
# %                             if any(keep.laserOn)
# %                                 keep.task_stop=keep.time(find(keep.laserOn,1,'last'));
# %                             else
#                                 warning('No task_stop time found! Defaulting to 20 minutes')
#                                 keep.task_stop=keep.task_start + 20 * 60; %Approximate
# %                             end
#                         elseif strcmp(keep.task,'10x30')
# %                             if any(keep.laserOn)
# %                                 keep.task_stop=keep.time(find(keep.laserOn,1,'last'));
# %                             else
#                                 warning('No task_stop time found! Defaulting to 30 minutes')
#                                 keep.task_stop=keep.task_start + 30 * 60; %Approximate 
# %                             end
#                         elseif strcmp(keep.task,'aversion')
# %                             if any(keep.laserOn)
# %                                 keep.task_stop=keep.time(find(keep.laserOn,1,'last'));
# %                             else
#                                 warning('No task_stop time found! Defaulting to 30 minutes')
#                                 keep.task_stop=keep.task_start + 30 * 60; %Approximate 
# %                             end.
#                         elseif strcmp(keep.task,'30min')
#                             keep.task_stop=30*60;
#                         elseif strcmp(keep.task,'60min') || strcmp(keep.task,'60min_psem') || strcmp(keep.task,'60min_saline')
#                             keep.task_stop=60*60;
#                         end
#                     end
#                     if strcmp(keep.zone,'zone_switch')
#                        %Task_start will have 4 times and task_stop will
#                        %have 4 times:
    
def stim_from_xlsx(df,pn):
    sheets=[key for key in df.keys()] #First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    header=get_header_from_rawdf(df)
    header_amount=int(header['Number of header lines:'])
    temp=df[sheets[1]]
    temp=temp.drop([i for i in range(0,(header_amount-3))])
    temp=temp.rename({col:temp[col][header_amount-3] for col in temp.columns},axis='columns')
    data=temp.drop(header_amount-3)
    
    stim_on=(data['Name']=='Is output 1 High') & ( data['Value']==1)
    stim_off=(data['Name']=='Is output 1 High') & ( data['Value']==0)
    stim={'on': data['Recording time'][stim_on].values,
          'off':data['Recording time'][stim_off].values,
          }
    if len(stim['on']) == 0:
        stim['on']=[np.nan]
        stim['off']=[np.nan]
    proto=str(pn).split(sep)[-3]
    stim['proto']=proto
    stim_dur=[]
    for i,onset in enumerate(stim['on']):
        if i< len(stim['off']):
            stim_dur.append(stim['off'][i]-onset)
        else: #If recording ended before stimulus shut off for some reason
            stim_dur.append(np.nan)
    stim['dur']=np.array(stim_dur)
    stim['amp_mw']=1 #By default, 1 mW
    stim['n']=len(stim['on'])
    if ('mw' in proto) or ('mW' in proto):
        stim['amp_mw']=int(proto.split('_')[-1].split('m')[0])
    if len(stim['dur']) > 0:        
        stim['mean_dur']=np.nanmean(stim['dur'])
    else:
        stim['mean_dur']=0
        # stim['dur']=[np.nan]
    return stim

def params_from_xlsx(df,pn):
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
    params: Dict
        Dictionary containing all relevant information about the given experiment,
        gleaned from header information in raw data df

    '''

    sheets=[key for key in df.keys()] #First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    nold=get_header_from_rawdf(df) #Header
    header_amount=int(nold['Number of header lines:'])
    temp=df[sheets[0]]
    str_pn=str(pn)
    anid=dataloc.path_to_animal_id(str_pn)
    
    #Check if there is any ambiguity about which mouse is in the file:
    nold_an=nold['Mouse ID'].values[0]
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
    light_method='Stim '
    if 'LED Dial' in nold:
        light_method='LED Dial'
    elif 'Dial' in nold:
        light_method='Dial'
    
    # Look for mouse sex information if available:
    if 'Sex' in nold:
        sex=nold['Sex'].values[0]
    elif 'Gender' in nold:
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
    if 'Stim #' in nold:
        stim_details=nold['Stim #'].values[0]
    elif 'LED Dial' in nold:
        stim_details=nold['LED Dial'].values[0]
        
    if 'Test' in nold:
        etho_exp=nold['Test'].values[0]
    else:
        etho_exp=np.nan
    params={
        'etho_animal_id': nold['Mouse ID'].values[0],
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
        'experimenter':str_pn.split(sep)[-2].split('_')[-1][0:2],
        'exp_room_number':room_num,
        'anid':anid,
        'folder_date':str_pn.split(sep)[-2].split('_')[-1][2:],
        'folder_anid':anid,
        'protocol':str_pn.split(sep)[-3],
        'side': str_pn.split(sep)[-4],
        'opsin_type':str_pn.split(sep)[-5],
        'cell_type':str_pn.split(sep)[-6],
        'da_state':str_pn.split(sep)[-7],
        'stim_area':str_pn.split(sep)[-8],
        'animal_id_mismatch':anid_mismatch,
        'possible_retrack':possible_retrack,
        'fs':29.97, # placeholder default frames / second but should confirm in each video if possible
        'task': str_pn.split(sep)[-3],
        'exp_start': 0,
        'exp_end':[],
        'task_start':[],
        'task_stop':[],
        }
    if 'zone' in params['protocol']:
        params['zone']='%s %s' % (params['protocol'].split('_')[0].capitalize(),
                                params['protocol'].split('_')[1])
    
    
    return params


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
        
    # data=pd.read_excel(pn,sheet_name=0,header = 38, na_values='-',engine='xlrd') #Assumes that header is always 38 lines--> is this true? can this be checked also?
    sheets=[key for key in df.keys()] #First sheet has tracking data, 2nd is hardware, 3rd is trial control signals 
    header=get_header_from_rawdf(df)
    header_amount=int(header['Number of header lines:'])
    temp=df[sheets[0]]
    temp=temp.drop([i for i in range(0,header_amount-3)])
    temp=temp.rename({col:temp[col][header_amount-3] for col in temp.columns},axis='columns')
    data=temp.drop(header_amount-3)
        
  
    #Check if first row has strings showing the units instead of data:
    first_val=data['Trial time'].values[0]
  
    if isinstance(first_val, str):
        print('\tDetected unnecessary row 0, dropping..')
        data.drop(axis=0, index=[header_amount-2], inplace=True)
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
                     'Mobility state(Immobile)':'im','Mobility state(Immobile)':'im',
                     'Mobility state(Mobile)':'m',
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

def add_amb_to_raw(raw,params,amb_thresh=2,amb_dur=0.5,im_thresh=1,im_dur=0.5):
    fs=params['fs']
    
    on,off=signal.thresh(raw['vel'],amb_thresh,sign='Pos')
    bouts=(off-on)/fs    
    amb=np.zeros(raw['vel'].shape,dtype=bool)
    im = amb
    m = np.ones(raw['vel'].shape,dtype=bool)
    for i,bout in enumerate(bouts):
        if bout > amb_dur:
            amb[on[i]:off[i]]=True
    
    im=raw['vel'] < im_thresh
    on,off=signal.thresh(im,0.5,sign='Pos')
    im = amb
    bouts=(off-on)/fs  
    for i,bout in enumerate(bouts):
        if bout > im_dur:
            im[on[i]:off[i]]=True
        
    m[im]=False  #Mobile = not immobile
    raw['im2']=im # 'im' is the immobility measure calculated in ethovision itself
    raw['m2']=m #m is the mobility measure
    raw['ambulation']=amb
    raw['fine_move']= (raw['ambulation']==False) & (raw['im']==False)
    return raw
    
    
    # keep.fine_move= ~keep.ambulation & ~keep.im; %Note: keep.im is the ethovision 'immobile' output