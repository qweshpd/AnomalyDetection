#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

_weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday","Sunday"]

_keys = ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday","Sunday", "Offday"]
_ntimes = 3

class WeeklyAnalysis(object):
    '''
    Analyze 1-D timeseries data based on weekliy information. Fit for those data
    with human behaviuors, i.e. weekly, and no trend. Currently support only 
    int times of hourly data.

    '''
    
    def __init__(self, data, holiday = None):
        '''
        Initialize.
        
        Parameters
        ----------       
        data : pandas.Series data with datetime-like index
            Data to be analyzed
        holiday : list of string
            Predefined holidays.
        '''
        
        self.data = data
        self.sdate = data.index[0].date()
        self.edate = data.index[-1].date()
        self.num = int(len(data) / ((self.edate - self.sdate).days + 1))
        self.freq = int(24 / self.num)
        self.lindex = pd.date_range(self.sdate, self.edate)
        self.sindex = pd.date_range(self.sdate, self.edate + datetime.timedelta(1),
                                    freq = str(self.freq) +'H')[:-1]
        self.dailydata = {}
        self.weekmodel = {}
        self.holiday = holiday
        self.columns = [('0' + str(i) + ':00')[-5:] for i\
                        in np.arange(self.num)*self.freq]
        
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
                   index = self.lindex.strftime('%Y-%m-%d'))
    
    def _get_dailydata(self, date):
        '''
        Get historical data on a specific weekday or weekend.    
        
        Parameters
        ----------       
        date : integer 
            Specific date, from 0 to 6 for weekday, or 7 for offday.
        '''
        if date == 7:
            # Offday = Saturday, Sunday and Holidays
            start = np.mod(5 - self.sdate.weekday(), 7) # Saturday start
            ind = np.arange(start, (self.edate - self.sdate).days + 1, 7)
            tmp = np.append(ind, ind + 1)
            for day in self.holiday:
                tmp = np.append(tmp, np.where(np.array(self.df.index) == day)[0])
            tmp.sort()
            self.dailydata['Offday'] = self.df.loc[self.df.index[tmp]]
        else:
            start = np.mod(date - self.sdate.weekday(), 7)
            ind = np.arange(start, (self.edate - self.sdate).days + 1, 7)
            dailydata = np.array(self.df)[ind]
            tmpind = list(np.array(self.df.index)[ind])
            tmpinddel = [tmpind.index(i) for i in tmpind if i in self.holiday]
            dailydata = np.delete(dailydata, tmpinddel, 0)
            ind = np.delete(ind, tmpinddel, 0)
            self.dailydata[_weekday[date]] = pd.DataFrame(dailydata, 
                                              index = self.df.index[ind],
                                              columns = self.columns)
        
    def plot_daily(self, *args):
        '''
        Plot historical daily data in matplotlib figure.    
        
        Parameters
        ----------       
        date : integer from 0 to 6
            Specific date.
        '''
        
        if args:
            dailydata = self.dailydata[args[0]]
        else:
            for day in _keys:
                self.plot_daily(day)
            return None
        
        def _onpick(event):
                print(event.artist)
                
        fig, ax = plt.subplots()
        for i in np.arange(dailydata.shape[0]):
            ax.plot(np.arange(self.num) + 1, np.array(dailydata)[i, :], 
                     label = dailydata.index[i], picker = True)
        fig.canvas.mpl_connect('pick_event', _onpick)    
        ax.legend().draggable()
        ax.set_ylim(0, np.ceil(1.04 * max(self.data)))
        pd.DataFrame(np.array(dailydata)).boxplot()
#        ax.set_xticks(np.arange(self.num) + 1)
        ax.set_xticklabels(self.columns)
        ax.set_title('Daily traffic on %s' % args[0])
        fig.show()
    
    def _get_dailymodel(self, *args):
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
        index = ['Ave', 'Max', 'Min', 'Std']
        
        if not args:
            daymodel = np.array([[],[],[],[]])
            col = []
            for date in _weekday:
                daymodel = np.hstack([daymodel, 
                              np.array(self._get_dailymodel(date))])
                col.append([date[:3] + i for i in self.columns])
            self.weekmodel['week'] = pd.DataFrame(daymodel,
                          columns = np.hstack(col), index = index)
            self._get_dailymodel('Offday')
        else:
            tmp = self.dailydata[args[0]]
            daymodel = pd.DataFrame(index = index)
            for time in self.columns:
                tmpdata = np.array(tmp[time])
                mmax, mmin = np.percentile(tmp[time], [75, 25])
                temp = tmpdata[np.where((tmpdata <= mmax  + 1.5 * (mmax - mmin)) & 
                                 (tmpdata >= mmin - 1.5 * (mmax - mmin)))]
                daymodel[time] = [temp.mean(), temp.max(), 
                                  temp.min(),  temp.std()]
            self.weekmodel[args[0]] = daymodel
            return daymodel

    def plot_weekmodel(self, *args):
        '''
        Plot regular wekkly model in matplotlib figure.
        
        Parameters
        ----------
        day : string, default = None
            Return the specific model of that day, or entire week model if 
            keep default.
        '''
        if args:
            if args[0] in _keys:
                daymodel = self.weekmodel[args[0]]
                title = args[0]
                xtick = self.columns
        else:
            daymodel = self.weekmodel['week']
            title = 'Weekly'
            xtick = []
            for day in _weekday:
                for time in self.columns:
                    xtick.append(day[:3] + time)
                    
        above = daymodel.loc['Ave'] + _ntimes * daymodel.loc['Std']
        below = daymodel.loc['Ave'] - _ntimes * daymodel.loc['Std']
        below[np.where(below < 0)[0]] = 0
            
        plt.figure()
        plt.plot(np.arange(daymodel.shape[1]), daymodel.loc['Ave'], 
                 '-', color = '#0072B2', label = 'Average')
        plt.fill_between(np.arange(daymodel.shape[1]), above, below, 
                         color = '#87CEEB', label = 'Confidence Inerval')
        plt.legend().draggable()
        plt.xticks(np.arange(daymodel.shape[1]), xtick, rotation = 40)
        plt.title(title + ' data model with confidence interval.')
        plt.grid()
        plt.show()

        
    def fit(self):
        '''
        Train model with data.   
        
        Parameters
        ----------
        hol: boolean, default = False
            If True, take holiday effect into consideration.
        '''
        self._get_df()
        
        for i in np.arange(8):
            self._get_dailydata(i)
            
        self._get_dailymodel()
    
    def _generate_model(self, sdate, edate, holiday = True):
        '''
        Generate data model within date range with freqency.   
        
        Parameters
        ----------
        sdate : string or datetime-like
            Left bound for generating dates
        edate : string or datetime-like
            Right bound for generating dates
        holiday ： boolean, or list of holidays' date, default True
            
        Returns
        ----------
        model : pandas.DataFrame
            Data model in the date range.
        '''
        s = sdate.weekday()
        e = edate.weekday()
        startday = sdate - datetime.timedelta(s)
        endday = edate + datetime.timedelta(6 - e)
        itera = int(((endday - startday).days + 1) / 7)
        model = np.array(self.weekmodel['week'])
        index = pd.date_range(sdate, edate + datetime.timedelta(1), 
                              freq = str(self.freq) +'H')[:-1]
        tmpind = pd.date_range(sdate, edate)
        
        for i in np.arange(itera - 1):
            model = np.hstack([model, np.array(self.weekmodel['week'])])
            
        if e == 6:
            model = model[:, self.num * s:]
        else:
            model = model[:, self.num * s:-((6 - e)*self.num)]
            
        if holiday is not False:
            if holiday is True:
                hol = self.holiday
            else:
                hol = holiday
                
            for i in np.arange(len(tmpind)):
                if tmpind[i].strftime('%Y-%m-%d') in hol:
                    model[:, self.num * i: self.num * i + self.num] =\
                                                np.array(self.weekmodel['Offday'])
        return pd.DataFrame(model, columns = index,
                     index = ['Ave', 'Max', 'Min', 'Std'])

        
    def detect(self, data = None, holiday = False, show = True):
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
        if data is None:
            data = self.data
        sdate = data.index[0].date()
        edate = data.index[-1].date()
        model = self._generate_model(sdate, edate, holiday = holiday)
        
        below = model.loc['Ave'] - _ntimes * model.loc['Std']
        above = model.loc['Ave'] + _ntimes * model.loc['Std']
        below[np.where(below < 0)[0]] = 0
        
        tsdata = np.array(data)
        ind = np.where((tsdata < below)|(tsdata > above))
        
        for i in self.data.index[ind]:
            print(i)
            
        if show:
            def _onpick(event):
                ind = event.ind
                print('Time: %s, Rate: %.3f' %
                      (model.columns[ind].strftime('%Y-%m-%d %H:%m-%s')[0],
                      np.take(tsdata, ind)))
                
            ind1 = np.where((tsdata >= below) & (tsdata <= above))
            if holiday is False:
                title = 'Data model without holidays.'
            else:
                title = 'Data model with customized holidays.'
                
            fig, ax = plt.subplots()
            ax.plot(model.columns, model.loc['Ave'], '-',
                      color = '#0072B2', label = 'Average')
            ax.fill_between(model.columns, above, below, 
                            color = '#87CEEB', label = 'Confidence Inerval')
            
            ax.scatter(model.columns[ind1], tsdata[ind1], c = 'k', 
                        label = 'Normal', picker = True)
            ax.scatter(model.columns[ind],  tsdata[ind],  c = 'r', 
                        label = 'Anomaly', picker = True)
            ax.set_title(title)
            ax.legend().draggable()       
            fig.canvas.mpl_connect('pick_event', _onpick)             
            ax.grid()
            plt.show()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        