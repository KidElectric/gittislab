#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:07:58 2021

@author: brian
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gittislab import signals, plots, behavior, dataloc, ethovision_tools, model
if ('COMPUTERNAME' in os.environ.keys()) \
    and (os.environ['COMPUTERNAME'] == 'DESKTOP-UR8URCE'):
        
    basepath = 'F:\\Users\\Gittis\\Dropbox\\Gittis Lab Data\\OptoBehavior\\'
else:
    basepath='/home/brian/Dropbox/Gittis Lab Data/OptoBehavior/'
# %%
# fn = Path('/home/brian/Dropbox/Gittis Lab Hardware/Laser Glow/laser_cal_sweeps/pwm_to_analog/power_control_sweep_0-255_arduino_multi_sweep_clean.csv')
fn = Path('/home/brian/Dropbox/Gittis Lab Hardware/Laser Glow/laser_cal_sweeps/pwm_to_TTL/power_control_sweep_0-255_arduino_TTL_multi_test_clean.csv')
df_cal=pd.read_csv(fn)
plt.figure()
y=df_cal.loc[:,'Power (W)'] *1000 # Put in mW
plt.plot(df_cal.loc[:,'Samples '],y)

#locate drop offs:
d=np.diff(y)
on,off=signals.thresh(d,-15,'Neg')
for o in off:
    plt.plot([o,o],[0,33],'--r')

onsets= [2] + off[0:-1]
offsets = on

newclips=[]
for o,f in zip(onsets,offsets):
    dd= f-o
    if dd > 62:
        o=f-62
    if dd < 62:
        o= o-(62-dd)
        
    t=y[o:f]
    print(len(t))
    newclips.append(t.values)

dat=np.stack(newclips)

plt.figure()
x=[x for x in range(0,255,round(255/62))]
x=x[2:]
i=0
for clip in dat:
    plt.plot(x,clip,label='%d' % i)
    i+=1
plt.legend()

ax=plots.mean_cont_plus_conf_array(x,dat.T,)
ax.set_ylabel('Blue laser output (mw)')
ax.set_xlabel('Arduino PWM level')

# %% 

def genPulseKey(ver,fs):    
    msPerSamp = 1000/fs
    if (ver == 0):
        timeKey={'BinPackStrt':[ 360, 100], #On dur, off dur
                  'BinPackStop':[ 160, 100],
                  'BitDur': [45,45]} #%Bit values are portrayed in 60ms length windows after BinPackStrt
    elif (ver >=1):
        timeKey={'BinPackStrt':[ 33*6, 66], #On dur, off dur
                  'BinPackStop':[ 33 *3, 66],
                  'BitDur': [25,66]} #%Bit values are portrayed 25ms length        
    sampKey={}
    for key in timeKey.keys():        
        pwSamp= [round(pwTime / msPerSamp) for pwTime in timeKey[key]]
        sampKey[key] = pwSamp
    return timeKey, sampKey

def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found
# %% Binary packet reading (AG6845_6_BI042021 etc.):
ex0=['exclude','Bad','GPe','bad','Broken', 'grooming',
 'Exclude','Other XLS']
exc=[ex0]
inc=[['AG','Str','A2A','Ai32','50x2',]]
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
df,meta=ethovision_tools.csv_load(pns[1],columns='All',method='raw' )
# # %% Binary packet from a test file:
# truth_pn=Path("/home/brian/Dropbox/Arduino/gittis/random_analog_leveler_v1/test_trial_338.csv")
# raw_pn=Path("/home/brian/Dropbox/Arduino/gittis/random_analog_leveler_v1/Raw data-bi_two_zone_rm216_v2-Trial   338.csv")
# df=pd.read_csv(raw_pn)
#%% Without interpolation:
correct_day=True
ver=0
evt = df['Binary packet info'].values.astype(int)
if correct_day == True:
    evt[1025:1029]=0 # Why are the stop and start events "Fused" here? bug?
    #evt[10257]=0
p=np.insert(evt,0,0)
dp=np.diff(p)
s=np.argwhere(dp==1)
ss = np.argwhere(dp==-1)
pwHigh = ss -s
pwLow= (ss[1:]-s[0:-1])
plt.figure(),plt.hist(pwHigh.flatten(),bins=range(0,13,1))

pack_per_trial= 2 #Expected number of binary packets per trial
fs = meta['fs'][0]

timeKey,sampKey=genPulseKey(ver,fs)  
packStartHigh, packStartLow=sampKey['BinPackStrt'] 
packStopHigh, packStopLow=sampKey['BinPackStop']
bitHigh, bitLow=sampKey['BitDur']

#Appoximate location of the first bit to read (measured from
#the beginning of the packStart pulse):
firstBitRead=packStartHigh+packStartLow  - 2

#Correct possible bit errors:

#PacketStart:
vals=[packStartHigh, packStopHigh,bitHigh]
for val in vals:
    errorInd= (pwHigh >= (val-2)) & (pwHigh <= (val+1))
    pwHigh[errorInd] = val

bitOn=s[pwHigh== bitHigh]
bitOff=ss[pwHigh==bitHigh]
bitLen=bitOff-bitOn
# evt[bitOn[bitLen==2]]=0
evt[bitOff[bitLen==2]-1]=0

packStart= s[pwHigh== packStartHigh]
packStop=s[pwHigh == packStopHigh]

output=[]
for start,stop in zip (packStart,packStop):
    bitRaw=np.zeros((16,1))
    bitDur=((stop-start - (firstBitRead -((bitHigh)/2)))/16)
    bitVector=[start + (round(firstBitRead + (bitDur*i))) for i in range(0,16)]
    i=0
    for a,b in zip(bitVector[0:-1],bitVector[1:]):
        bitRaw[i] = any(evt[a:b])
        i+=1
    bitRaw=np.flipud(bitRaw)
    byte_str=''
    for b in bitRaw:
        byte_str=byte_str+str(b[0].astype(int))
    output.append(int(byte_str,2))
o=np.array(output).reshape((50,2))
plt.figure()
plt.plot(o[:,0])

o[37,0]=38
# %% Examine packet
trial=8
i= (trial-1) * 2
start=packStart[i]
stop=packStop[i]

fullStop=ss[pwHigh == packStopHigh][i]
plt.figure()
plt.plot(evt[start:fullStop])
# plt.plot(orig_evt[start:stop],'k')
plt.plot(evt[start:stop],'r')

bitVector=[(firstBitRead + (bitDur*i)) for i in range(0,16)]
# bitVector=[ (round(firstBitRead + (bitDur*i))) for i in range(0,16)]
i=0
for a,b in zip(bitVector[0:-1],bitVector[1:]):
    plt.plot([a,b],[0.5,0.5])
    
plt.xlabel('Video Frames')
plt.ylabel('Signal from Arduino')
plt.title('Trial %s' % trial)
# %% Go from trial light value to mW (using dat above)
x=[x for x in range(0,255,round(255/62))]
x=x[2:]
mean_power=np.mean(dat,axis=0)
f = np.polyfit(x,mean_power,deg=2)
p = np.poly1d(f)

ax=plots.mean_cont_plus_conf_array(x,dat.T,)
ax.set_ylabel('Blue laser output (mw)')
ax.set_xlabel('Arduino PWM level')
plt.sca(ax)
plt.plot(x,p(x),'r')

ex0=['exclude','Bad','GPe','bad','Broken', 'grooming',
 'Exclude','Other XLS']
exc=[ex0]
inc=[['AG','Str','A2A','Ai32','50x2',]]
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
raw,meta=ethovision_tools.csv_load(pns[1],columns='All',method='preproc' )

meta['stim_dur'] = 2 #Overwrite default
meta['stim_off'] = meta['stim_on'] + 2
percentage = lambda x: (np.nansum(x)/len(x))*100
raw['m'] = ~raw['im']
m_clip= behavior.stim_clip_grab(raw,meta,y_col='m', 
                               stim_dur=2,
                               baseline = 2,
                               summarization_fun=percentage)

y=m_clip['disc'][:,1]
x=p(o[:,1])
plt.figure()
plt.plot(x,y,'ok')

#Bin per mW and average:
ym=[]
for i in range(0,8,1):
    ind = (x >= i) & (x < (i+1))
    mbin= np.mean(y[ind])
    ym.append(mbin)
xm=[i+0.5 for i in range(0,8,1)]
plt.plot(xm,ym,'or')
plt.plot(0,np.mean(m_clip['disc'][:,0]),'*b')
# xf = np.append(x,np.zeros(len(m_clip['disc'][:,0])))
# yf = np.append(y,m_clip['disc'][:,0])
po,pc = model.fit_sigmoid(x,y)

xs=[i/100 for i in range(25,800,1)]
ys=model.sigmoid(xs, po[0], po[1], po[2],po[3])
plt.plot(xs,ys,'b')
plt.xlabel('Power (mW)')
plt.ylabel('% Time mobile')

# %% Percent Time Immobile

m_clip= behavior.stim_clip_grab(raw,meta,y_col='im', 
                               stim_dur=2,
                               baseline = 2,
                               summarization_fun=percentage)

y=m_clip['disc'][:,1]
x=p(o[:,1])
plt.figure()
plt.plot(x,y,'ok')

#Bin per mW and average:
ym=[]
for i in range(0,8,1):
    ind = (x >= i) & (x < (i+1))
    mbin= np.mean(y[ind])
    ym.append(mbin)
xm=[i+0.5 for i in range(0,8,1)]
plt.plot(xm,ym,'or')
plt.plot(0,np.mean(m_clip['disc'][:,0]),'*b')
# xf = np.append(x,np.zeros(len(m_clip['disc'][:,0])))
# yf = np.append(y,m_clip['disc'][:,0])
po,pc = model.fit_sigmoid(x,y)

xs=[i/100 for i in range(25,800,1)]
ys=model.sigmoid(xs, po[0], po[1], po[2],po[3])
plt.plot(xs,ys,'b')
plt.xlabel('Power (mW)')
plt.ylabel('% Time Immobile')

# %% Ambulation bout param:
m_clip = behavior.bout_analyze(raw,meta,'amb',stim_dur=2,
                 min_bout_dur_s=0.5,
                 min_bout_spacing_s=0.1,
                 use_dlc=False,
                 calc_meander = False)

y=m_clip['speed'][:,1]
y[np.isnan(y)] = 0
x=p(o[:,1])
plt.figure()
plt.plot(x,y,'ok')

#Bin per mW and average:
ym=[]
for i in range(0,8,1):
    ind = (x >= i) & (x < (i+1))
    mbin= np.mean(y[ind])
    ym.append(mbin)
xm=[i+0.5 for i in range(0,8,1)]
plt.plot(xm,ym,'or')
plt.plot(0,np.mean(m_clip['speed'][:,0]),'*b')
# xf = np.append(x,np.zeros(len(m_clip['disc'][:,0])))
# yf = np.append(y,m_clip['disc'][:,0])
po,pc = model.fit_sigmoid(x,y)

xs=[i/100 for i in range(25,800,1)]
ys=model.sigmoid(xs, po[0], po[1], po[2],po[3])
plt.plot(xs,ys,'b')
plt.xlabel('Power (mW)')
plt.ylabel('Amb. bout speed')

# %% same as above but with speed:
    
m_clip= behavior.stim_clip_grab(raw,meta,y_col='vel', 
                               stim_dur=2,
                               baseline = 2,
                               summarization_fun=np.mean)

y=(m_clip['disc'][:,1])
x=p(o[:,1])
plt.figure()
plt.plot(x,y,'ok')

#Bin per mW and average:
ym=[]
for i in range(0,8,1):
    ind = (x >= i) & (x < (i+1))
    mbin= np.mean(y[ind])
    ym.append(mbin)
xm=[i+0.5 for i in range(0,8,1)]
plt.plot(xm,ym,'-or')
plt.plot(0,np.mean(m_clip['disc'][:,0]),'*b')

# po,pc = model.fit_sigmoid(x,y)
# xs=[i/100 for i in range(25,800,1)]
# ys=model.sigmoid(xs, po[0], po[1], po[2],po[3])
# plt.plot(xs,ys,'b')
plt.xlabel('Power (mW)')
plt.ylabel('Speed (cm/s)')
#%% With interpolation DOES NOT WORK WELL.

# evt = df['Binary packet info'].values.astype(int)
# evt[1025:1029]=0 # Why are the stop and start events "Fused" here? bug?
# p=np.insert(evt,0,0)
# dp=np.diff(p)
# s=np.argwhere(dp==1)
# ss = np.argwhere(dp==-1)
# pwHigh = ss -s
# pwLow= (ss[1:]-s[0:-1])
# # plt.figure(),plt.hist(pwHigh.flatten(),bins=range(0,13,1))

# pack_per_trial= 2 #Expected number of binary packets per trial
# fs = meta['fs'][0]
# ver=0
# timeKey,sampKey=genPulseKey(ver,fs)  
# packStartHigh, packStartLow=sampKey['BinPackStrt'] 
# packStopHigh, packStopLow=sampKey['BinPackStop']
# bitHigh, bitLow=sampKey['BitDur']

# #Appoximate location of the first bit to read (measured from
# #the beginning of the packStart pulse):


# #Correct possible bit errors:

# #PacketStart:
# vals=[packStartHigh, packStopHigh,bitHigh]
# for val in vals:
#     errorInd= (pwHigh >= (val-2)) & (pwHigh <= (val+1))
#     pwHigh[errorInd] = val

# # bitOn=s[pwHigh== bitHigh]
# # bitOff=ss[pwHigh==bitHigh]
# # bitLen=bitOff-bitOn
# # # evt[bitOn[bitLen==2]]=0
# # evt[bitOff[bitLen==2]-1]=0

# packStart= s[pwHigh== packStartHigh]
# packStop=s[pwHigh == packStopHigh]
# interp_amnt=11 #1 = no interpolation

# firstBitRead=packStartHigh+packStartLow -2
# first_bit= 152 #(firstBitRead-1)*interp_amnt

# output=[]
# for start,stop in zip (packStart,packStop):
#     bitRaw=np.zeros((16,1))
    
#     # Take packet and interpolate it:
#     packet = evt[start:stop]
#     interp_pack=(np.interp(np.arange(0, len(packet), 1/interp_amnt), 
#                            np.arange(0, len(packet)), packet)>0).astype(int)
#     bitDur=(len(interp_pack) - first_bit)/16
#     bitVector=[round(first_bit + (bitDur*i)) for i in range(0,16)]
#     i=0
#     for a,b in zip(bitVector[0:-1],bitVector[1:]):
#         ave=round((a+b)/2)
#         bitRaw[i] = interp_pack[ave]
#         # bitRaw[i] = any(interp_pack[(ave):round(ave+(b-ave)/2)])
#         # bitRaw[i] = any(interp_pack[(ave+3):b])
#         i+=1
#     bitRaw=np.flipud(bitRaw)
#     byte_str=''
#     for b in bitRaw:
#         byte_str=byte_str+str(b[0].astype(int))
#     output.append(int(byte_str,2))
# o=np.array(output).reshape((50,2))
# plt.figure()
# plt.plot(o[:,0])


# # %% Examine interp packet
# trial= 1
# i= (trial-1) * 2
# start=packStart[i]
# stop=packStop[i]
# packet = evt[start:stop]
# interp_pack=(np.interp(np.arange(0, len(packet), 1/interp_amnt), 
#                        np.arange(0, len(packet)), packet)>0).astype(int)


# plt.figure()
# # first_bit=(firstBitRead)*interp_amnt
# plt.plot(interp_pack,'k')
# bitDur=(len(interp_pack) - first_bit)/16
# bitVector=[round(first_bit + (bitDur*i)) for i in range(0,16)]
# i=0
# for a,b in zip(bitVector[0:-1],bitVector[1:]):
#     plt.plot([a,b],[0.5,0.5])
#     plt.plot([round((a+b)/2),b],[0.75,0.75],'r')
#     # plt.plot(b,0.5,'*g')
