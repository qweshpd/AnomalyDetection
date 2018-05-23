#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 15, 6

_eachday = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday","Sunday", "Offday"]
_ntimes = 3
_index = ['Ave', 'Max', 'Min', 'Std']
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
    
    def __init__(self, holiday = [], freq = 2):
        '''
        Initialize the model with holidays.
        
        Parameters
        ----------       
        holiday: list of string, YYYY-MM-DD
            Predefined holidays.
        '''
        self.freq = freq
        self.num = int(24/self.freq)
        self.columns = [('0' + str(int(i)) + ':00')[-5:] for i\
                in np.linspace(0, 24, self.num + 1)[:-1]]
        self.holiday = holiday
        self.dailydata = {}
        self.dailymodel = {}
        for day in _eachday:
            self.dailydata[day] = pd.DataFrame(columns = ['value', 'sec'])
            self.dailymodel[day] = pd.DataFrame(columns = self.columns,
                           index = _index)
        
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
            self.dailydata[date].loc[day_of_data] = np.hstack([slice_of_data, sec])
            
        return self.dailydata
        
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
    
    def decompose(self, data, holiday = []):
        '''
        Decompose data into everyday.
        
        Parameters
        ----------       
        holiday: list of string, YYYY-MM-DD
            Predefined holidays.
        '''
        self.data = data
        _ = self._extract_day()
        tmp = np.array([[],[],[],[]])
        col = []
        for day in _eachday[:-1]:
            tmp = np.hstack([tmp, np.array(self._build_model(day))])
            col.append([day[:3] + i for i in self.columns])
        self.dailymodel['week'] = pd.DataFrame(tmp, columns = np.hstack(col), 
                                               index = _index)
        _ = self._build_model('Offday')

            
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
        xtic = pd.date_range('2018-01-01', '2018-01-02', freq = 'S')[:-1]
        xaxis = [xtic[int(i)] for i in data[:, 1]]
        fig, ax = plt.subplots()
        ax.scatter(xaxis, data[:, 0], picker = True)
        fig.canvas.mpl_connect('pick_event', _onpick)    
#        ax.legend().draggable()
#        ax.set_ylim(0, np.ceil(1.04 * np.max(np.array(self.data))))
        ax.set_xlim(xtic[0], xtic[-1])
#        pd.DataFrame(np.array(dailydata)).boxplot()
#        ax.set_xticklabels(pd.date_range('2019-01-01', '2019-01-02', freq = 'S')[:-1])
        ax.set_title('Daily traffic on %s' % _eachday[args[0]])
        fig.show()
        
        
    def _build_model(self, day):
        '''
        Build daily model based on data.  
               
        Parameters
        ----------       
        day : string
            Day to build daily model.
        '''
        tmpdict = {}
        if day == 'Offday':
            tmpdata = np.vstack([np.array(self.dailydata[i]) for i in _eachday[-3:]])
        else:
            tmpdata = np.array(self.dailydata[day])
        
        for time in self.columns:
            tmpdict[time] = []
        
        for i in np.arange(tmpdata.shape[0]):
            ind = int(np.round(tmpdata[i, 1] / 3600 / self.freq))
            if ind == self.num:
                ind -= 1
            tmpdict[self.columns[ind]].append(tmpdata[i, 0])
            
        for time in self.columns:
            time_tmp = np.array(tmpdict[time])
            mmax, mmin = np.percentile(tmpdict[time], [75, 25])
            temp = time_tmp[np.where((time_tmp <= mmax + 1.5 * (mmax - mmin))\
                                  & (time_tmp >= mmin - 1.5 * (mmax - mmin)))]
            self.dailymodel[day][time] = [temp.mean(), temp.max(), 
                                          temp.min(),  temp.std()]
            
        return self.dailymodel[day]
        
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
            if args[0] in _eachday:
                daymodel = self.dailymodel[args[0]]
                title = args[0]
                xtick = self.columns
        else:
            daymodel = self.dailymodel['week']
            title = 'Weekly'
            xtick = []
            for day in _eachday:
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
        startday = sdate - timedelta(s)
        endday = edate + timedelta(6 - e)
        itera = int(((endday - startday).days + 1) / 7)
        model = np.array(self.dailymodel['week'])
        index = pd.date_range(sdate, edate + timedelta(1), 
                              freq = str(self.freq) +'H')[:-1]
        tmpind = list(pd.date_range(sdate, edate).strftime('%Y-%m-%d'))
        
        for i in np.arange(itera - 1):
            model = np.hstack([model, np.array(self.dailymodel['week'])])
            
        if e == 6:
            model = model[:, self.num * s:]
        else:
            model = model[:, self.num * s:-((6 - e)*self.num)]
            
        if holiday is not False:
            if holiday is True:
                hol = self.holiday
            else:
                hol = holiday

            for day in hol:
                if day in tmpind:
                    i = tmpind.index(day)
                    model[:, self.num * i: self.num * i + self.num] =\
                                    np.array(self.dailymodel['Offday'])
                                        
        return pd.DataFrame(model, columns = index, index = _index)        
        
        
    def detect(self, data = None, holiday = False, where = 'both', show = True):
        '''
        Fit trained model to data, and get anomaly data point.   
        
        Parameters
        ----------
        data :  pandas.Series data
            Data to test.
        hol: boolean, default = False
            If True, take holiday effect into consideration.
        where : string, ['above', 'below', 'both']
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
        indabove = np.where(tsdata > above)[0]
        indbelow = np.where(tsdata < below)[0]
        
        printext = []
        
        if where == 'both':
            printext.append('Above Normal Range:\n')
            if not len(indabove):
                printext.append('None.\n')
            else:
                for i in indabove:
                    printext.append(data.index[i].strftime('%Y-%m-%d %H:%m:%S'))
            printext.append('\nBelow Normal Range:\n')
            if not len(indbelow):
                printext.append('None.\n')
            else:
                for i in indbelow:
                    printext.append(data.index[i].strftime('%Y-%m-%d %H:%m:%S'))
        elif where == 'above':
            for i in indabove:
                printext.append(data.index[i].strftime('%Y-%m-%d %H:%m:%S'))
        elif where == 'below':
            for i in indbelow:
                printext.append(data.index[i].strftime('%Y-%m-%d %H:%m:%S'))
            
        for i in printext:
            print(i)
            
        if show:
            def _onpick(event):
                ind = event.ind
                print('\nTime: %s, Rate: %.3f' %
                      (data.index[ind].strftime('%Y-%m-%d %H:%m:%S')[0],
                      tsdata[ind]))
                
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
            
            if where == 'both':
                ax.scatter(model.columns[indabove], tsdata[indabove],  c = 'r', 
                            label = 'Above Normal', picker = True)
                ax.scatter(model.columns[indbelow], tsdata[indbelow],  c = 'pink', 
                            label = 'Below Normal', picker = True)
            elif where == 'above':
                ax.scatter(model.columns[indabove], tsdata[indabove],  c = 'r', 
                            label = 'Anormal', picker = True)
            elif where == 'below':
                ax.scatter(model.columns[indbelow], tsdata[indbelow],  c = 'r', 
                            label = 'Anormal', picker = True)
            
            ax.set_title(title)
            ax.legend().draggable()       
            fig.canvas.mpl_connect('pick_event', _onpick)             
            ax.grid()
            plt.show()
        
        return indabove, indbelow
        
        
        
        
        