#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:05:49 2021

@author: brian
"""

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

def genPulseKey(ver,fs):    
    msPerSamp = 1000/fs
    if (ver < 1):
        timeKey={'BinPackStrt':[ 360, 100], #On dur, off dur
                  'BinPackStop':[ 160, 100],
                  'BitDur': [45,45]} #%Bit values are portrayed in 60ms length windows after BinPackStrt
    elif (ver <= 2):
        timeKey={'BinPackStrt':[ 33*6, 66], #On dur, off dur
                  'BinPackStop':[ 33 *3, 66],
                  'BitDur': [25,66]} #%Bit values are portrayed 25ms length   
    elif (ver <=3 ):
        timeKey={'BinPackStrt':[ 33*10, 66], #On dur, off dur
                  'BinPackStop':[ 33 *5, 66],
                  'BitDur': [34,66]} #%Bit values are portrayed 25m
        
    elif (ver <=4 ):
        timeKey={'BinPackStrt':[ 33*10, 66], #On dur, off dur
                  'BinPackStop':[ 33 *5, 66],
                  'BitDur': [66,66]} #%Bit values are portrayed 25m
        
    sampKey={}
    for key in timeKey.keys():        
        pwSamp= [round(pwTime / msPerSamp) for pwTime in timeKey[key]]
        sampKey[key] = pwSamp
    return timeKey, sampKey


# %% Binary packet reading:
ex0=['exclude','Bad','GPe','bad','Broken', 'grooming',
 'Exclude','Other XLS']
exc=[ex0]
inc=[['AG','Str','A2A','Ai32','50x2_hm4di_cno',]]
pns=dataloc.raw_csv(basepath,inc[0],ex0)
if not isinstance(pns,list):
    pns=[pns]
use = pns[4]
df,meta=ethovision_tools.csv_load(use,columns='All',method='raw' )

truth_pn=dataloc.gen_paths_recurse(use.parent,[],[],filetype='pwm_output*.csv')
# truth_pn=pns[0].parent.joinpath('pwm_output_trial_354.csv')
pwm_df=pd.read_csv(truth_pn)`

#%% Without interpolation:
correct_day=False
ver=3
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
    errorInd= (pwHigh >= (val-1)) & (pwHigh <= (val+1))
    pwHigh[errorInd] = val

# bitOn=s[pwHigh== bitHigh]
# bitOff=ss[pwHigh==bitHigh]
# bitLen=bitOff-bitOn
# # evt[bitOn[bitLen==2]]=0
# evt[bitOff[bitLen==2]-1]=0

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

cor_t = np.sum(pwm_df['Trial'].values == o[:,0])
cor_pwm = np.sum(pwm_df['PWM'].values == o[:,1])
print('%d/%d correct trial bytes, %d/%d correct PWM bytes' % (cor_t,o.shape[0],
                                                              cor_pwm,o.shape[0]))

# %% Examine packet
trial=17
ii= (trial-1) * 2
start=packStart[ii]
stop=packStop[ii]

fullStop=ss[pwHigh == packStopHigh][ii]
plt.figure()
plt.plot(evt[start:fullStop])
# plt.plot(orig_evt[start:stop],'k')
plt.plot(evt[start:stop],'r')

# bitVector=[(firstBitRead + (bitDur*i)) for i in range(0,16)]
bitVector=[ (round(firstBitRead + (bitDur*i))) for i in range(0,16)]
i=0
for a,b in zip(bitVector[0:-1],bitVector[1:]):
    plt.plot([a,b],[0.5,0.5])
    
i=0
start = packStart[ii]
bitVector = bitVector + start
bitRaw=np.zeros((16,1))
for a,b in zip(bitVector[0:-1],bitVector[1:]):
    bitRaw[i] = any(evt[a:b])
    i+=1
bitRaw=np.flipud(bitRaw)
byte_str=''
for b in bitRaw:
    byte_str=byte_str+str(b[0].astype(int))
plt.xlabel('Video Frames')
plt.ylabel('Signal from Arduino')
plt.title('Trial %s, byte: %s ( = %d)' % (trial,byte_str, int(byte_str,2)))

# %%  How many correct / incorrect?

cor_t = np.sum(pwm_df['Trial'].values == o[:,0])
cor_pwm = np.sum(pwm_df['PWM'].values == o[:,1])
print('%d/%d correct trial bytes, %d/%d correct PWM bytes' % (cor_t,o.shape[0],
                                                              cor_pwm,o.shape[0]))
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
