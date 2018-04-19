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
            Start and end date of data.
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
        self.num = int(24/freq)
        self.columns = [('0' + str(i) + ':00')[-2:] for i in np.arange(self.num)*freq]
        self._get_df()
        
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
        tmp = np.array(self.data).reshape(len(index), self.num)
        self.df = pd.DataFrame(tmp, columns = self.columns, 
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
        
        start = np.mod(date - self.index[0].weekday(), 7)
        ind = np.arange(start, (self.index[1] - self.index[0]).days, 7)
        dailydata = np.array(self.df)[ind]
        tmpind = list(np.array(self.df.index)[ind])
        tmpinddel = [tmpind.index(i) for i in tmpind if i in self.holiday]
        dailydata = np.delete(dailydata, tmpinddel, 0)
        inde = np.delete(np.array(self.df.index)[ind], tmpinddel, 0)
        ind = np.delete(ind, tmpinddel, 0)

        if show:
            plt.figure()
            for i in np.arange(dailydata.shape[0]):
                plt.plot(np.linspace(1, self.num, self.num), 
                         dailydata[i, :], label = inde[i])
            plt.legend(loc = 'best')
            pd.DataFrame(dailydata).boxplot()
            plt.title('Daily traffic on %s' % _weekday[date])
            plt.show()
        
        return pd.DataFrame(dailydata, index = self.df.index[ind],
                            columns = self.columns)
    
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
        for i in np.arange(7):
            setattr(self, _weekday[i], self._get_daily(self, i))
        
        
        