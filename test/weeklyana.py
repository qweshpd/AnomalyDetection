#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime

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
        self.sdate = index[0]
        self.edate = index[1]
        self.lindex = pd.date_range(index[0], index[1])
        self.sindex = pd.date_range(index[0], index[1] + datetime.timedelta(1), 
                                    freq = str(freq) +'H')
        self.freq = freq
        self.holiday = holiday
        self.num = int(24 / freq)
        self.columns = [('0' + str(i) + ':00')[-5:] for i in np.arange(self.num)*freq]
        self._get_df()
        
    def _get_df(self):
        '''
        Transform pandas.Series data into pandas.DataFrame format.

        Returns
        ----------
        df : pandas.DataFrame
            Formatted data.
        '''
        tmp = np.array(self.data).reshape(len(self.lindex), self.num)
        self.df = pd.DataFrame(tmp, columns = self.columns, 
                   index = [date.strftime('%Y-%m-%d') for date in self.lindex])
        
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
        
        start = np.mod(date - self.sdate.weekday(), 7)
        ind = np.arange(start, (self.edate - self.sdate).days + 1, 7)
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
            plt.legend().draggable()
            plt.ylim(0, int(str(int(str(int(max(self.data)))[:2]) + 2) \
                            + str(int(max(self.data)))[2:]))
            pd.DataFrame(dailydata).boxplot()
            plt.title('Daily traffic on %s' % _weekday[date])
            plt.show()
        
        return pd.DataFrame(dailydata, index = self.df.index[ind],
                            columns = self.columns)
    
    def weekfit(self):
        '''
        Transfer data into each weekday and holiday.   
        
        Parameters
        ----------
        show : boolean
            If True, plot daily data in matplotlib figure.
        
        '''
        for i in np.arange(7):
            setattr(self, _weekday[i], self._get_daily(i))
            
        # Offday = Saturday, Sunday and Holidays
        start = np.mod(5 - self.sdate.weekday(), 7) # Saturday start
        ind = np.arange(start, (self.edate - self.sdate).days + 1, 7)
        tmp = np.append(ind, ind + 1)
        for day in self.holiday:
            tmp = np.append(tmp, np.where(np.array(self.df.index) == day)[0])
        tmp.sort()
        self.Offday = self.df.loc[self.df.index[tmp]]
    
    def _somedaymodel(self, day):
        '''
        Build daily model.   
        
        Parameters
        ----------
        day : string, default = None
            Return the specific model of that day, or entire week model if 
            keep default.
            
        Returns
        ----------
        daymodel : pandas.DataFrame
            Model of a specfic day.
        '''
        tmp = getattr(self, day)
        daymodel = pd.DataFrame(index = ['Ave', 'Max', 'Min', 'Std'])
        for time in self.columns:
            tmpdata = np.array(tmp[time])
            mmax, mmin = np.percentile(tmp[time], [75, 25])
            temp = tmpdata[np.where((tmpdata <= mmax  + 1.5 * (mmax - mmin)) & 
                             (tmpdata >= mmin - 1.5 * (mmax - mmin)))]
            daymodel[time] = [temp.mean(), temp.max(), 
                              temp.min(),  temp.std()]
        self.daymodel[day] = daymodel
        return daymodel
        
        
    def dailymodel(self, show = False):
        '''
        Build basic model based on historical data.   
        
        Parameters
        ----------
        day : string, default = None
            Return the specific model of that day, or entire week model if 
            keep default.
        show : boolean
            If True, plot daily data in matplotlib figure.
            
        Returns
        ----------
        daymodel : pandas.DataFrame
            
        '''
        # fit before modeling
        if not hasattr(self, 'Offday'):
            self.weekfit()
        
        self.daymodel = {}

        daymodel = np.array([[],[],[],[]])
        col = []
        for date in _weekday:
            daymodel = np.hstack([daymodel, 
                                 np.array(self._somedaymodel(date))])
            col.append([date[:3] + i for i in self.columns])
        daymodel = pd.DataFrame(daymodel, columns = np.hstack(col),
                                index = ['Ave', 'Max', 'Min', 'Std'])
        _ = self._somedaymodel('Offday')
        self.weekmodel = daymodel

        if show:
            plt.figure()
            plt.plot(np.arange(daymodel.shape[1]), daymodel.loc['Ave'], 
                     '-', color = '#0072B2', label = 'Average')
            plt.fill_between(np.arange(daymodel.shape[1]), 
                             daymodel.loc['Ave'] + 3 * daymodel.loc['Std'], 
                             daymodel.loc['Ave'] - 3 * daymodel.loc['Std'], 
                             color = '#87CEEB', label = 'Confidence Inerval')
            plt.legend().draggable()
            plt.title('Average weekly data model with confidence interval.')                 
            plt.grid()
            plt.show()

        return daymodel
        
    def fitmodel(self, data = None, hol = False, show = True):
        '''
        Fit trained model to data, and get anomaly data point.   
        
        Parameters
        ----------
        data :  pandas.Series data
            Data to test.
        hol: boolean, default = False
            If True, take holiday effect into consideration.
        show : boolean, default = True
            If True, plot daily data in matplotlib figure.
            
        Returns
        ----------
        anomalies : numpy.array of strings
            Anomalies date and time.
        '''
        if not hasattr(self, 'daymodel'):
            self.dailymodel()
            
        if data is None:
            tsdata = np.array(self.data)
        else:
            tsdata = np.array(data)

        s = self.sdate.weekday()
        e = self.edate.weekday()
        startday = self.sdate - datetime.timedelta(s)
        endday = self.edate + datetime.timedelta(7 - e)
        itera = int((endday - startday).days / 7)
        model = np.array(self.dailymodel())
        
        for i in np.arange(itera - 1):
            model = np.hstack((model, np.array(self.dailymodel())))
            
        model = model[:, self.num * s:-((6 - e)*self.num)]
        
        title = 'without holidays.'
        
        if hol:
            title = 'with holidays.'
            for i in np.arange(len(self.lindex)):
                if self.lindex[i].strftime('%Y-%m-%d') in self.holiday:
                    model[:, 12 * i: 12 * i + self.num] = np.array(self.daymodel['Offday'])
        
        self.model = model

        if show:
            below = self.model[0, :] - 3 * self.model[3, :]
            above = self.model[0, :] + 3 * self.model[3, :]
            plt.figure()
            plt.plot(np.arange(self.num * len(self.lindex)), self.model[0, :], 
                     '-', color = '#0072B2', label = 'Average')
            plt.fill_between(np.arange(self.num * len(self.lindex)), above, below, 
                            color = '#87CEEB', label = 'Confidence Inerval')
            
            ind1 = np.where((tsdata >= below) & (tsdata <= above))
            ind2 = np.where( (tsdata < below) | (tsdata > above) )
            
#            df = pd.DataFrame()
#            df['data'] = tsdata
#            df['time'] = pd.date_range(self.index[0].strftime('%Y-%m-%d'), 
#                (self.index[1] + datetime.timedelta(1)).strftime('%Y-%m-%d'),
#                freq = '2H')[:-1]
#            
#            df[ind1].plot(kind = 'scatter', x = 'time', y = 'data', c = 'b')
#            df[ind2].plot(kind = 'scatter', x = 'time', y = 'data', c = 'r')
            plt.scatter(ind1, tsdata[ind1], c = 'k', label = 'Normal')
            plt.scatter(ind2, tsdata[ind2], c = 'r', label = 'Anomaly')
            plt.xticks(self.num * np.arange(len(self.lindex)), self.df.index)
            plt.title('Data model ' + title)
            plt.legend().draggable()                    
            plt.grid()
            plt.show()   
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        