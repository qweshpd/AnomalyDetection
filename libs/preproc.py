#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Collection of necessary preprocessing method.
'''

import datetime
import pandas as pd
import numpy as np

def getdtime(ymd):
    '''
    Convert YYYY-MM-DD to datetime.datetime format.
    '''
    y, m, d  = map(int, ymd.split('-'))
    return datetime.datetime(y, m, d, 0, 0, 0)

def normalize(data, method):
    if method == '01':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    if method == '-11':
        return (data - np.mean(data)) / (np.max(data) - np.min(data))