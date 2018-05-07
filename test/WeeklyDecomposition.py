#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

_eachday = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday","Sunday"]


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
            self.dailydata[day] = pd.DataFrame(columns = ['time', 'value', 'sec'])
        
    def _extract_day(self):
        '''
        Decompose data into eachday.
        
        Parameters
        ----------       
        data : pandas.Series data with datetime-like index
            Data to be analyzed.
        holiday : list of string, YYYY-MM-DD
            Predefined holidays.
        '''

        for dayts in self.data:
            if dayts.date().strftime('%Y-%m-%d') in self.holiday:
                self.dailydata['holiday'].append(dayts)
            else:
                date = _eachday[dayts.weekday()]
                self.dailydata[date].append(dayts)
    
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
        self._extract_day()
        
        
            
        