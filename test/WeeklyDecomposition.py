#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

_eachday = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday","Sunday", "Holiday"]


#%%
class WeeklyDecomposition(object):
    '''
    Decompose 1-D timeseries data based on weekliy information, 
    regardless frequency.
    
    Parameters
    ----------       
    data : pandas.Series data with datetime-like index
        Data to be analyzed.
    holiday : list of string, YYYY-MM-DD
        Predefined holidays.
    '''
    
    def __init__(self, data):

        self.data = data
        self.holiday = []
        self.dailydata = {}
        for day in _eachday:
            self.dailydata[day] = pd.DataFrame(columns = ['value', 'sec'])
        
    def _extract_day(self):
        '''
        Decompose data into eachday.
        
        Parameters
        ----------       
        data : pandas.Series data with datetime-like index
            Data to be analyzed.
        '''
        data_index = self.data.index
        for day_of_data in data_index:
            
            if day_of_data.date().strftime('%Y-%m-%d') in self.holiday:
                date_ind = 7
            else:
                date_ind = day_of_data.weekday()
            
            sec = self._convert_time(day_of_data)
            date = _eachday[date_ind]
            slice_of_data = np.array(self.data.loc[day_of_data])
            self.dailydata[date].loc[day_of_data] = [slice_of_data, sec]
        
        
    def _convert_time(self, daytime):
        '''
        Convert a datetime value to real second time.
        
        Parameters
        ----------       
        daytime : datetime.datetime instance
            Datetime to be converted.
        '''
        time0 = daytime.date().strftime('%Y-%m-%d')
        datetime0 = datetime.strptime(time0, '%Y-%m-%d')
        ind = (daytime - datetime0).seconds
        return ind
    
    def decompose(self, holiday = []):
        '''
        Decompose data into everyday.
        
        Parameters
        ----------       
        holiday: list of string, YYYY-MM-DD
            Predefined holidays.
        '''
        self.holiday = holiday
        self._extract_day()
        
    
    def plot_daily(self, *args):
        '''
        Plot historical daily data in matplotlib figure.    
        
        Parameters
        ----------       
        date : integer from 0 to 7
            Specific date.
        '''
        
        if args:
            if args[0] in np.arange(8):
                dailydata = self.dailydata[_eachday[args[0]]]
            else:
                for i in np.arange(8):
                    self.plot_daily(i)
                    return None
        else:
            for i in np.arange(8):
                self.plot_daily(i)
                return None
        
        def _onpick(event):
            time = dailydata.index[event.ind]
            print(time)
                
        data = np.array(dailydata)
        fig, ax = plt.subplots()
        ax.scatter(data[:,1], data[:, 0], picker = True)
        fig.canvas.mpl_connect('pick_event', _onpick)    
        ax.legend().draggable()
        ax.set_ylim(0, np.ceil(1.04 * max(self.data)))
#        pd.DataFrame(np.array(dailydata)).boxplot()
#        ax.set_xticklabels(self.columns)
        ax.set_title('Daily traffic on %s' % _eachday[args[0]])
        fig.show()
        
        
            
        