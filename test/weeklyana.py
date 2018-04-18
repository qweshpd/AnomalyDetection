#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


class WeeklyAnalysis(object):
    '''
    Analyze 1-D timeseries data based on weekliy information.

    '''
    
    def __init__(self, data, index, freq = 2, holiday = None):
        '''
        Initialize.
        
        Parameters
        ----------       
        data : 1D array like 
            Data to be analyzed
        index : 1D array like DatetimeIndex
            Timestamp of data. Must be of the same size.
        freq : integer
            Frequency of data collected. Currently support only int times of 
            hourly data.
        holiday : list of string
            Predefined holidays.
        '''
        
        self.data = data
        self.index = index
        self.freq = freq
        self.holiday = holiday

    def _get_df(self):
        '''
        Transform pandas.Series data into pandas.DataFrame format.
    
        Parameters
        ----------       
    
        
        Returns
        ----------
        
        '''
        sdate = self.index[0].strftime('%Y-%m-%d')
        edate = self.index[-1].strftime('%Y-%m-%d')
        index = pd.date_range(sdate, edate)
        num = int(24/self.freq)
        col = ['hr' + str(i * self.freq) for i in range(num)]
        tmp = np.array(self.data).reshape(len(index), num)
        df = pd.DataFrame(tmp, columns = col, 
                          index = [date.strftime('%Y-%m-%d') for date in index])
        
        return df
#%%



