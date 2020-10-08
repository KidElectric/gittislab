"""
    dataloc contains functions for quickly navigating a formatted file strcture
    
    gen_paths_recurse(basepath,inc=[],exc = [],filetype = None):
        gen_paths_recurse takes a file path name to search for directories that
        include the strings in list "inc," and exclude strings in the list "exc" -> (optional).
             
    path_to_fn(basepath,inc=[],exc= [],filetype = None, include_fn=True):
       Using the input methods of gen_paths_recurse(),
       return a new file name that replaces file separation ('/') with underscore '_',
       and existing underscores '_' with hyphens '-' to allow a reversable process (fn_to_path)
             
             
"""
import os
from pathlib import Path
if os.name == 'posix':
    sep='/'
else:
    sep='\\'
    
def gen_paths_recurse(basepath,inc=[],exc = [],filetype = None):
    """ gen_paths_recurse takes a file path name to search for directories that
        include the strings in list "inc," and exclude strings in the list "exc" -> (optional).
        filetype - If string containing file extension is included in filetype, will return path
                   to matching file types in paths that meet inc & exc criteria

        pn="F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\"
        str_to_contain=["GPe","A2A","ChR2","10x30","AG3461_2"]
        str_to_exc=["0p25mw","0p5mw","Left","Right","exclude"] #Not totally necessary here, but just for example
        filetype=".mpg"
        >>> out= gen_paths_recurse(pn,str_to_contain,str_to_exc,filetype)
        >>> print(out)
        >>> WindowsPath('F:/Users/Gittis/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/A2A/ChR2/Bilateral/10x30/AG3461_2_BI041219/Raw_AG3461_2_BI041219.mat')
   """
    basepath=Path(basepath)
    str_to_contain=inc
    str_to_contain.append("AG") #Required for all mouse-level folders in the project.
    str_to_exclude=exc
    str_to_exclude.append("exclude")
    exc_allf=['~lock','._']
    output=[]
    for dirpath, dirnames, files in os.walk(basepath):
        cont_all=[i in dirpath for i in str_to_contain] #Boolean list of desired target strings
        allhit=sum(cont_all)==len(str_to_contain)
        ex_all=[i in dirpath for i in str_to_exclude] #Boolean list of undesired target strings
        fa=sum(ex_all) > 0
        if allhit==True and fa==False:
            if filetype != None:
                if '*' in filetype: #If wildcard present
                    filetype = filetype.split('*')
                elif not isinstance(filetype,list):
                    filetype=[filetype]
                for f in files:
                    cont_allf=[i in f for i in filetype]
                    file_hit=sum(cont_allf)==len(filetype)
                    file_ex=sum([i in f for i in exc_allf]) > 0
                    if file_hit==True and file_ex==False:
                        output.append(Path(dirpath,f))
            else:
                output.append(Path(dirpath))
    if len(output)==1:
        output=output[0]
    return output

def common_paths(use_labels=[]):
    """
    Input
        use_labels : list
            DESCRIPTION: 
                Optional input list of labels to limit output to just those criteria
                See: labels output of function without any input for full list.
    Return an up-to-date list of commonly used experimental conditions and paths
        to be used for gen_paths_recurse() and other functions
        
    Returns
    Commonly-used lists of 
    -------
    basepath : str
        directory where gen_paths_recurse should begin
    inc : list of list of str
        File parts to include (input to gen_paths_recurse)
    exc : list of list of str
        File parts to exclude (input to gen_paths_recurse)
    labels : list of str
        readable list of condition types corresponding to each list in inc,exc

    """
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior'
    labels=['GPe-CAG-Arch',
        'Str-A2a-ChR2',
        'GPe-PV-Arch',
        'GPe-PV-ChR2',
        'GPe-PV-EYFP',
        'Str-A2a-Ai32',
        'GPe-Lhx6i-PVoff-Halo',
        'GPe-Lhx6i-PVoff-ChR2',
        'GPe-Npas1-Arch',
        'GPe-Npas1-ChR2',
        'GPe-A2a-0.25mw-ChR2',
        'GPe-A2a-1.0mw-ChR2']
    
    temp_inc=[['GPe','Arch','CAG','10x30','Bilateral'],
        ['Str','10x30','A2A','ChR2','Bilateral'],
        ['GPe','Arch','PV','10x30','Bilateral'],
        ['GPe','ChR2','PV','10x30','Bilateral'],
        ['GPe','EYFP_green_light','PV','10x30','Bilateral'],
        ['Str','10x30','A2A','Ai32','Bilateral'],
        ['GPe','Halo','Lhx6','10x30','Bilateral'],
        ['GPe','ChR2','Lhx6','10x30','Bilateral'],
        ['GPe','Arch','Npas','10x30','Bilateral'],
        ['GPe','ChR2','Npas','10x30','Bilateral'],
        ['GPe','A2A','10x','0p25mw','Bilateral'],
        ['GPe','A2A','10x','Bilateral']]
    
    temp_exc=[['SNr','Str','AG3351_2'], #Excluding 1 good file because it bugs that I need to figure out later! eg AG3351_2_
        ['GPe','0p25mw','Gad2'],
        ['SNr','Str','AG3396_2'], #Excluding 1 good file because it bugs that I need to figure out later!
        ['SNr','Str','Lhx6iCre'],
        ['SNr','Str'],
        ['GPe','0p25mw','Gad2'],
        ['SNr'],
        ['SNr','AG4898_3'], #Excluding 1 good file because it bugs that I need to figure out later!
        ['SNr'],
        ['SNr'],
        ['EYFP','duty','GAD2'],
        ['EYFP','mw','duty','GAD2','AG3773_5']]  #Excluding 1 good file because it bugs that I need to figure out later!
    inc=[]
    exc=[]
    if any(use_labels):
        use_ind= [labels.index(lab) for lab in use_labels]
        inc=[temp_inc[i] for i in use_ind]
        exc=[temp_exc[i] for i in use_ind]
        labels=use_labels
    else:
        inc=temp_inc
        exc=temp_exc
    if len(inc)==1:
        inc=inc[0]
        exc=exc[0]
    return basepath,inc,exc,labels

def path_to_fn(basepath,inc=[],exc= [],filetype = None, include_fn=True):
    """ Using the input methods of gen_paths_recurse(),
    return a new file name that replaces file separation ('/') with underscore '_',
    and existing underscores '_' with hyphens '-' to allow a reversable process (fn_to_path)
    if include_fn = True, filename will be included in output
    See: gen_paths_recurse() for further help.
    >>>
    """
    locs= gen_paths_recurse(basepath,inc,exc,filetype)
    newfns=[]
    for loc in locs:
        parts=str(loc).split(sep)
        newstr=''
        temp=0
        for i,part in enumerate(parts):
            if part not in basepath:
                if temp == 0:
                    temp=i
                if ' ' in part:
                    part=part.replace(' ','')
                if '_' in part:
                    part=part.replace('_','-')
                if i > temp:
                    part = '_' + part
                if '.' in part and include_fn == False:
                    part=filetype
                newstr += '%s' % part
        newfns.append(newstr)
    return newfns

def path_to_description(path):
    '''
    path_to_description(path) - Take a path str or Path object and convert last 7 folders
        into a string
        e.g. input = '/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/GPe/Naive/CAG/Arch/Bilateral/10x30/AG3351_2_JS051118/Raw_AG3351_2_JS051118.mat'
        output = 'GPe_Naive_CAG_Arch_Bilateral_10x30_AG3351_2_JS051118'
    Parameters
    ----------
    path : Str or str-like (Path)
        Path to a file to be summraized.

    Returns
    -------
    path_str : Str
        Descriptive str with last 7 folders of path separated by underscores.

    '''
    
    if not isinstance(path,str):
        str_pn=str(path)
    else:
        str_pn=path
        
    path_parts=str_pn.split(sep)[-8:-1]
    path_str=''
    for part in path_parts:
        path_str += part + '_'
    path_str=path_str[0:-1]
    return path_str

def fn_to_path(basepath,file_list,max_depth = 6,iteration = 0):
    """ Take a list of filenames as generated by path_to_fn() and turn them
    into 2 lists: paths and filenames, including basepath. File name underscores
    are turned to file separation ('/') and '-' to underscores.

    Example:
    local_vid='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior'
    example_fns=['GPe_Naive_PV_ChR2_Bilateral_zone-1_AG5304-1-UZ030520DLC_resnet50_psc_analyzeMay1shuffle1_1000000.csv',
                 'Str_Naive_D1_Arch_Bilateral_10x30-3mW_AG4924-1-BI102319DLC_resnet50_psc_analyzeMay1shuffle1_1000000.csv']
    max_depth=6 #default / optional
    iter = 10 #default / optional
    >>> pn,fn = dataloc.fn_to_path(local_vid,example_fns,max_depth,iter)
    """
    newfns=[]
    newpns=[]
    for fn in file_list:
        parts=fn.split('_') #Split fn using underscore
        pathstr=''
        if iteration > 0:
            fnstr='iteration_%d_' % iteration
        else:
            fnstr=''
        for i,part in enumerate(parts):
            if i > max_depth:
                #Remove 'DLC' from the name:
                pathstr=pathstr[:-3]

                #Use rest as fn:
                for p in parts[i:]:
                    fnstr += '%s_' % p
                fnstr = fnstr[:-1]

                break
            if '-' in part:
                part=part.replace('-','_')
            part = sep + part
            pathstr += '%s' % part
        newfns.append(fnstr)
        newpns.append(basepath + pathstr)
    return newpns,newfns

def video(basepath,inc=[],exc=[]):
    """ Using the input file path "basepath,"
    return the .mpg video locations for all mice that
    match the criteria of "inc" list of strings and "exc" string list (optional)
    See: gen_paths_recurse() for further help.
    >>>
    """
    return gen_paths_recurse(basepath,inc,exc,filetype = '.mpg')

def rawmat(basepath,inc=[],exc = []):
    """ Using the input file path "basepath,"
    return the .mpg video locations for all mice that
    match the criteria of "inc" list of strings and "exc" string list (optional)
    
    NOTE: simplest use case is to only include a specific directory in 'basepath'
        and rawmat will return the path to .mat file in that directory if present
    See: gen_paths_recurse() for further help.
    >>>
    """
    keep_path=gen_paths_recurse(basepath,inc,exc,filetype = 'Raw*.mat')

    return keep_path

def rawxlsx(basepath,inc=[],exc = []):
    """ Using the input file path "basepath,"
    return the .xlsx locations for all mice that
    match the criteria of "inc" list of strings and "exc" string list (optional)
    
    NOTE: simplest use case is to only include a specific directory in 'basepath'
        and rawxlsx will return the path to .xlsx file in that directory if present
    See: gen_paths_recurse() for further help.
    >>>
    """
    keep_path=gen_paths_recurse(basepath,inc,exc,filetype = 'Raw*.xlsx')
    return keep_path

def rawh5(basepath,inc=[],exc = []):
    """ Using the input file path "basepath,"
    return the .h5 locations for all mice that
    match the criteria of "inc" list of strings and "exc" string list (optional)
    
    NOTE: simplest use case is to only include a specific directory in 'basepath'
        and rawh5 will return the path to .h5 file in that directory if present
    See: gen_paths_recurse() for further help.
    >>>
    """
    keep_path=gen_paths_recurse(basepath,inc,exc,filetype = 'Raw*.h5')
    return keep_path

def dlc_h5(basepath,inc=[],exc = []):
    """ Using the input file path "basepath,"
    return the dlc_analyze.h5 deeplabct file locations for all mice that
    match the criteria of "inc" list of strings and "exc" string list (optional)
    
    NOTE: simplest use case is to only include a specific directory in 'basepath'
        and dlc_h5 will return the path to .h5 file in that directory if present
    See: gen_paths_recurse() for further help.
    >>>
    """
    keep_path=gen_paths_recurse(basepath,inc,exc,filetype = '*dlc_analyze.h5')
    return keep_path

def path_to_animal_id(pn):
    an= '%s_%s' % tuple(pn.split(sep)[-2].split('_')[0:2])
    return an

def path_to_rawfn(pn):
    '''
        Generate generic raw file name from a FILE path pn
        
        output: raw (str)
            in format: Raw_CAGE_ANIMALID_EXPDATE,
            
        input: path str:
        
            
        e.g. if pn= ~/10x30/AG6151_3_CS090720/Raw data-BI_two_zone_closed_loop_v3_retrack-Trial     1.xlsx'
         raw = Raw_AG6151_3_CS090720
    '''
    if not isinstance(pn,str):
        pn=str(pn)
    raw= 'Raw_%s' % (pn.split(sep)[-2])
    return raw

def full_exp_dict(ver = 0):
    """ Generate full dictionary of all experiments run
    for BI GPe inhibition project
    based on MATLAB function "opto_filepath_params_gen_v1()
    if ~exist('spec','var')
    spec={'pn'  'stim_area'  'da_state'  'cell_type'  'opsin_type'  'side'  'protocol'};

    """
    if ver == 0: # Updated through 4/20/2020
        full_exp = {'stim_area':['Str','GPe','GPi','GPe_and_Str', 'SNr','SNr_and_Str','GPe_and_SNr'],
                    'da_state':['Naive'],
                    'cell_type': ['PV', 'hSyn','CAG','PVflip_off_Lhx6iCre_on',
                                  'Npas','A2A','D1','Lhx6iCre','CAG_D1','GAD2_A2A','D2SP','A2a_Gad2',
                                  'PV_PV','D2_D1','Gad2','D1_D1','D2'],
                    'virus_type':['Arch','Ai32','Ai35','Halo','ChR2','Ai35',
                                  'EYFP_hSyn_ChR2','Caspase_hSyn_ChR2','EYFP_hSyn_ChR2','Arch_caspase',
                                  'EYFP_green_light','EYFP_blue_light','Ctl','NoVirus','D2SP-ChR2',
                                   'Caspase_ChR2','PSAM_PSEM','PSAM_saline','Chrimson_ChR2','ChR2_Chrimson',
                                   'ChR2_Caspase','ChR2_mCherry','Chrimson','stGtACR2','Chrimson_stGtACR2',
                                   'Ai35_Ai35','Psam4GlyR'],
                    'side':['Bilateral','Left','Right'],
                    'protocol':['zone_1','zone_2','10x10','10x30','5x30','10x30_0p5mw',
                                'zone_1_0p25mw','10x30_0p25mw','pp30_base','5x30_0p25mw','5x30_2mw',
                                'zone_1_d1r_0p25mw','zone_1_d1r','20min_10Hz','20min_4Hz','10x30_3mW',
                                'zone_1_0p5mw','10x10_0p25mw','10x10_0p5mw','80min','zone_1_max_0p25mw',
                                '10x10_max','10x10_green','10x10_max','10x10_2mw','zone_switch',
                                '60min_psem','60min_saline','zone_2_0p25mw','10x10_30mW',
                                'zone_1_30mW','5x30_30mW'],
                    }
    return full_exp
