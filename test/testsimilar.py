#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.stats
import numpy as np
from libs.process import detect_onset

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def dissimilar(twosier, slot = 50, show = True):
    '''
    Compare each part of two time series data with a fixed sliding window, 
    and those below the overall Pearson correlation coefficient are detected 
    as dissimilar.
    
    Parameters
    ----------       
    twosier : two 1-D array-like
         Time series data to compare.
    Returns
    -------
    ind : 1-D array-like
        Indices of dissimilar parts.

    '''
    seir1, seir2 = twosier
    assert len(seir1) == len(seir2)
    
    nsample = len(seir1)
    
    sp1 = [seir1[i: i + slot] for i in np.arange(nsample - slot + 1)]
    sp2 = [seir2[i: i + slot] for i in np.arange(nsample - slot + 1)]
    
    res = []
    for i in np.arange(len(sp1)):
        res.append(scipy.stats.pearsonr(sp1[i], sp2[i]))
    
    ind = detect_onset(-np.array(res)[:, 0], n_above = 5,
                       threshold = -scipy.stats.pearsonr(seir1, seir2)[0])   
    
#    if len(ind):
#        for tmpind in ind:
#            if np.diff(tmpind) >= slot:
#                tmpind[0] += slot
#        tmpind[-1][1] == nsample - slot
        
    if show:
        plt.figure()
        plt.subplot(221)
        plt.plot(seir1)
        plt.subplot(222)
        plt.plot(seir2)
        plt.subplot(223)
        plt.plot(1-np.array(res)[:, 0])
        plt.subplot(224)
        plt.plot(seir1)
        plt.plot(seir2)
        for oneind in ind:
            plt.axvline(x=oneind[0], color='b', lw=1, ls='--')
            plt.axvline(x=oneind[1], color='b', lw=1, ls='--')
            
    return ind

