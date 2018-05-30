#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.stats
import numpy as np
from libs.process import detect_onset

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

def dissimilar(seir1, seir2):
    assert len(seir1) == len(seir2)
    
    nsample = len(seir1)
    num = 50
    
    sp1 = [seir1[i: i + num] for i in np.arange(nsample - num + 1)]
    sp2 = [seir2[i: i + num] for i in np.arange(nsample - num + 1)]
    
    res = []
    for i in np.arange(len(sp1)):
        res.append(scipy.stats.pearsonr(sp1[i], sp2[i]))
    
    ind = detect_onset(-np.array(res)[:, 0], n_above = 5,
                       threshold = -scipy.stats.pearsonr(seir1, seir2)[0])   
    return ind
