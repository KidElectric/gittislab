# Collection of functions related to using LED or LASER stimulation
def plot_laser_cal(file_path):    
    # import os
    # from gittislab import signal # sys.path.append('/home/brian/Dropbox/Python')
    # from gittislab import dataloc
    from matplotlib import pyplot as plt
    # import numpy as np
    import pandas as pd
    fig,ax=plt.subplots(1,1)
    df = pd.read_excel(file_path,engine="odf")
    ax.scatter(df['dial'],df['cable_power'])
    ax.set_xlabel('Dial Value')
    ax.set_ylabel('Raw Output (No Cannula) mW')
file_path='/home/brian/Dropbox/Gittis Lab Hardware/Laser Glow/laser_cal_080320.ods'    
plot_laser_cal(file_path)