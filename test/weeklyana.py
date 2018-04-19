#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

_weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
             "Saturday","Sunday"]

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
        df : pandas.DataFrame
            Formatted data.
        '''
        index = pd.date_range(self.index[0].strftime('%Y-%m-%d'), 
                              self.index[1].strftime('%Y-%m-%d'))
        num = int(24/self.freq)
        col = ['hr' + str(i * self.freq) for i in range(num)]
        tmp = np.array(self.data).reshape(len(index), num)
        self.df = pd.DataFrame(tmp, columns = col, 
                          index = [date.strftime('%Y-%m-%d') for date in index])
        
        return self.df
    
    def _get_daily(self, date, show = False):
        '''
        Get historical data on a specific weekday or weekend.    
        
        Parameters
        ----------       
        date : integer from 0 to 6
            Specific date.
        show : boolean
            If True, plot daily data in matplotlib figure.
        
        Returns
        ----------
        daydf : numpy.array
            Formatted daily data.
        '''
        self._get_df()
        start = np.mod(date - self.index[0].weekday(), 7)
        ind = np.arange(start, (self.index[1] - self.index[0]).days, 7)
        dailydata = np.array(self.df)[ind]
        
        if show:
            num = int(24/self.freq)
            plt.figure()
            for i in np.arange(len(ind)):
                plt.plot(np.linspace(1, num, num), dailydata[i, :])
            plt.boxplot(dailydata)
            plt.title('Daily traffic on %s' % _weekday[date])
        return dailydata
    
    def weekfit(self, show = False):
        '''
        Get historical data on a specific weekday or weekend.    
        
        Parameters
        ----------
        show : boolean
            If True, plot daily data in matplotlib figure.
        
        Returns
        ----------
        daydf : pandas.DataFrame
            Formatted data.
        '''
                
