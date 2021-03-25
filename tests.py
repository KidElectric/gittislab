#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:44:29 2021

@author: brian
"""
import numpy as np
from gittislab import signal

def eval_array(obs,known):
    assess=[]
    for i,on in enumerate(known):
        if i > (len(obs)-1):
            assess.append(False)
        else:
            assess.append(on == obs[i])
    return np.array(assess)
            
def signal_thresh():
    '''
    Test thresholding function in gittislab.signal.thresh().
    '''
    y=np.array([1,0,0,1,1,-4.6,1,0,1])
    thresh = 0
    print('Signal = %s' % str(y))
    print('Threshold = %1.2f' % thresh)
    
    n,f=signal.thresh(y,thresh,sign='Pos')
    
    #Detect above threshold, test indexed onsets
    pass_test = True
    print('Detections above threshold, onsets:')
    correct = [0,3,6,8]
    assess = eval_array(n,correct)
    if any(assess == False):
        print('\tFAILED.') 
        pass_test = False
    else:
        print('\tPASSED')
    print('\t\tCorrect: Onset Ind = %s' % str(correct))
    print('\t\tObserved: Onset Ind =%s' % str(n)) 
    
    #Detect above threshold, test indexed offsets
    print('Detections above threshold, offsets:')
    correct = [1, 5, 7, 8]
    assess = eval_array(f,correct)
    if any(assess == False):
        print('\tFAILED.') 
        pass_test = False
    else:
        print('\tPASSED')
    print('\t\tCorrect: Onset Ind = %s' % str(correct))
    print('\t\tObserved: Onset Ind =%s' % str(f))

    
    #Detect below threshold, test indexed onsets
    y=-y
    print('\nSignal = %s' % str(y))
    print('Threshold = %1.2f' % thresh)
    n,f=signal.thresh(y,thresh,sign='Neg')
    
    print('Detections below threshold, onsets:')
    correct = [0,3,6,8]
    assess = eval_array(n,correct)
    if any(assess == False):
        print('\tFAILED.')    
        pass_test = False
    else:
        print('\tPASSED')        
    print('\t\tCorrect: Onset Ind = %s' % str(correct))
    print('\t\tObserved: Onset Ind =%s' % str(n))
    
    #Detect below threshold, test indexed offsets
    print('Detections above threshold, offsets:')
    correct = [1, 5, 7, 8]
    assess = eval_array(f,correct)
    if any(assess == False):
        print('\tFAILED.')     
        pass_test = False
    else:
        print('\tPASSED')
    print('\t\tCorrect: Onset Ind = %s' % str(correct))
    print('\t\tObserved: Onset Ind =%s' % str(f))
    
    if pass_test == False:
        print('OVERALL: TEST FAILED')
    else:
        print('OVERALL: TEST PASSED')
    return pass_test