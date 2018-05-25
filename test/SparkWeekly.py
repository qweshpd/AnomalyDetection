#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import matplotlib.pylab as plt
plt.rcParams['figure.figsize'] = 15, 6

_eachday = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday","Sunday", "Offday"]
_ntimes = 3
_index = ['Ave', 'Max', 'Min']

class SparkWeekly(object):
    '''
    Decompose 1-D timeseries data based on weekliy information with spark.
    
    Parameters
    ----------       
    data : pandas.Series data with datetime-like index
        Data to be analyzed.
    holiday : list of string, YYYY-MM-DD
        Predefined holidays.
    '''     
    def __init__(self, holiday = [], ignoday = [], freq = 2, filt = None, sc = None, merge = False):
        '''
        Initialize the model with holidays.
        
        Parameters
        ----------       
        holiday: list of string, YYYY-MM-DD
            Predefined holidays.
        ignoday: list of string, YYYY-MM-DD
            Days not considered into model.
        freq: int
            The expected data frequency.
        filt: function
            The function of how data model will be build.
        sc: pyspark.SparkContext
            The SparkContext, required.
        merge: boolean, default = False
            True if want to build all weekday into one model.
        '''
        assert sc != None, "Missing SparkContext"
        
        if filt is None:
            filt = lambda x: [np.array(x).mean(), np.array(x).mean() + 3 * np.array(x).std(),\
                              np.array(x).mean() - 3 * np.array(x).std()]
        self.filt = filt
        self.sc = sc
        self.merge = merge
        self.holiday = holiday
        self.ignoday = ignoday
        self.freq = freq
        self.num = int(24/self.freq)
        self.columns = [('0' + str(int(i)))[-2:] for i in np.linspace(0, 24, self.num + 1)[:-1]]
        self.dailydata = {}
        self.dailymodel = {}
        for day in _eachday:
            self.dailydata[day] = pd.DataFrame(columns = ['data', 'ind', 'time'])
            self.dailymodel[day] = pd.DataFrame(columns = self.columns,
                           index = _index)

    def _extract_day(self):
        '''
        Decompose data into eachday.
        
        Parameters
        ----------       
        merge: boolean, default = False
            True if want to build all weekday into one model.
        '''
        sc = self.sc
        holiday = self.holiday
        ignoday = self.ignoday
        data = self.data
        freq = self.freq
        tmp_data = sc.parallelize(np.vstack((list(data.index), list(data))).T)\
                     .filter(lambda x: x[0].strftime('%Y-%m-%d') not in ignoday)
        
        # Build model for weekday
        reg_day = tmp_data.filter(lambda x: x[0].strftime('%Y-%m-%d') not in holiday)       
        if self.merge is False:
            reg_day_data = reg_day\
                      .map(lambda x:(str(x[0].weekday()) + ('0' + str(int(x[0].hour/freq)))[-2:], [x[1], x[0]]))\
                      .reduceByKey(lambda x,y:np.vstack((np.array(x),np.array(y)))).sortByKey()
            
            for i in np.arange(7):        
                tmp_data = reg_day_data.filter(lambda x: x[0].startswith(str(i)))\
                     .map(lambda x:(x[0][1:], x[1]))
                self.dailydata[_eachday[i]] = tmp_data.collectAsMap()
        else:
            reg_day_data = reg_day.filter(lambda x: x[0].weekday() < 5)\
                      .map(lambda x:(('0' + str(int(x[0].hour/freq)))[-2:], [x[1], x[0]]))\
                      .reduceByKey(lambda x,y:np.vstack((np.array(x),np.array(y)))).sortByKey()
            self.dailydata['Busday'] = tmp_data.collectAsMap()
    
        # Build model for offday 
        off_day = tmp_data.filter(lambda x: (x[0].strftime('%Y-%m-%d') in holiday) or (x[0].weekday() > 4))
        off_day_data = off_day\
                  .map(lambda x:(('0' + str(int(x[0].hour/freq)))[-2:], [x[1], x[0]]))\
                  .reduceByKey(lambda x,y:np.vstack((np.array(x),np.array(y)))).sortByKey()
        
        self.dailydata['Offday'] = off_day_data.map(lambda x:(x[0], x[1])).collectAsMap()
  
        return reg_day_data, off_day_data
        
    def fit(self, data):
        '''
        Decompose data into everyday.
        
        Parameters
        ----------       

        '''
        filt = self.filt
        self.data = data
        reg_day_data, off_day_data = self._extract_day()
        
        wk = pd.DataFrame([])
        col = []
        
        if self.merge is False:
            for i in np.arange(7):
                rdd = reg_day_data.filter(lambda x:x[0].startswith(str(i)))\
                                          .map(lambda x:(x[0][1:], x[1][:, 0]))
                tmpmodel = self._build_model(rdd, filt)                         
                col.append([_eachday[i][:3] + k for k in self.columns])
                wk = wk.T.append(tmpmodel.T).T
                self.dailymodel[_eachday[i]] = tmpmodel
                offrdd = off_day_data.map(lambda x:(x[0], x[1][:, 0]))        
                self.dailymodel['Offday'] = self._build_model(offrdd, filt) 
        else:
            busrdd = reg_day_data.map(lambda x: (x[0], x[1][:, 0]))
            tmpmodel = self._build_model(busrdd, filt)
            for i in np.arange(5):
                self.dailymodel[_eachday[i]] = self._build_model(busrdd, filt)
                col.append([_eachday[i][:3] + k for k in self.columns])
                wk = wk.T.append(tmpmodel.T).T
            offrdd = off_day_data.map(lambda x:(x[0], x[1][:, 0]))        
            offmodel = self._build_model(offrdd, filt)
            self.dailymodel['Offday'] = offmodel
            for i in [5, 6]:
                self.dailymodel[_eachday[i]] = offmodel
                col.append([_eachday[i][:3] + k for k in self.columns])
                wk = wk.T.append(tmpmodel.T).T
        wk.columns = np.hstack(col)
        self.dailymodel['week'] = wk
            
    def _build_model(self, RDDdata, filt):
        '''
        Build daily model based on data.  
               
        Parameters
        ----------       
        RDDday : spark.RDD object
            Day to build daily model.
        '''
        model = RDDdata.map(lambda x: (x[0], filt(x[1])))
        modeldf = pd.DataFrame(model.collectAsMap(), 
                               columns=model.keys().collect(), 
                               index=_index)
        modeldf.columns = self.columns
        return modeldf     
            
    def plot_daily(self, *args):
        '''
        Plot historical daily data in matplotlib figure.    
        
        Parameters
        ----------       
        date : integer from 0 to 7
            Specific date.
        '''
        
        if args and (args[0] in np.arange(8)):
            datadic = self.dailydata[_eachday[args[0]]]
        else:
            for i in np.arange(8):
                self.plot_daily(i)
            return

        data = np.array([[], [], []]).T
        for xtic in datadic:
            tp = np.hstack([datadic[xtic],\
                            self.freq*int(xtic)*np.ones((datadic[xtic].shape[0], 1))])
            data = np.vstack((data, tp))
            
        def _onpick(event):
            time = data[event.ind, 1]
            print(time)
            
        fig, ax = plt.subplots()
        ax.scatter(data[:, 2], data[:, 0], picker = True)
        fig.canvas.mpl_connect('pick_event', _onpick)    
#        ax.legend().draggable()
#        ax.set_xticklabels(self.columns)
        ax.set_title('Daily traffic on %s' % _eachday[args[0]])
        fig.show()
       
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
                    
        above = daymodel.loc['Max']
        below = daymodel.loc['Min']
            
        plt.figure()
        plt.plot(np.arange(self.num), daymodel.loc['Ave'], 
                 '-', color = '#0072B2', label = 'Average')
        plt.fill_between(np.arange(self.num), above, below, 
                         color = '#87CEEB', label = 'Confidence Inerval')
        plt.legend().draggable()
        plt.xticks(np.arange(self.num), xtick, rotation = 40)
        plt.title(title + ' data model with confidence interval.', fontsize = 20)
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
        
#%%        
    def detect(self, data = None, holiday = True, where = 'both', show = True):
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
        abv : 1D array-like
            Indices of anomalies above.
        bel : 1D array-like
            Indices of anomalies below.
        nor: 1D array-like
            Indices of normal data point.
        '''
        mode = self.dailymodel
        def _mapfunc(data):
            value = data[1]
            time = data[0]
            mmax = np.array(mode[_eachday[time.weekday()]]\
                       .loc[['Max'], [('0' + str(int(time.hour/freq)))[-2:]]])[0][0]
            mmin = np.array(mode[_eachday[time.weekday()]]\
                       .loc[['Min'], [('0' + str(int(time.hour/freq)))[-2:]]])[0][0]
            if value > mmax:
                color = 'r'
            elif value < mmin:
                color = 'pink'
            else:
                color = 'k'
                
            return (time, value, color)
            
            
        if data is None:
            data = self.data
        sc = self.sc
        freq = self.freq
        sdate = data.index[0].date()
        edate = data.index[-1].date()
        model = self._generate_model(sdate, edate, holiday = holiday)
        res = sc.parallelize(np.vstack((list(data.index), list(data))).T)\
                .map(lambda x:_mapfunc(x))
        result = np.vstack(res.collect())
        
        abv = np.where(result[:, 2] == 'r')
        bel = np.where(result[:, 2] == 'pink')
        nor = np.where(result[:, 2] == 'k')

        filtdict = {'above': ['r'], 'below': ['pink']}

        if where == 'both':
            where = ['above', 'below']
        for col in list(where):
            print(col + ' Normal Range:\n')
            pp = result[result[:, 2] == filtdict[col]]
            for date in pp[:, 0]:
                print(date.strftime('%Y-%m-%d %H:%m:%S'))
            
        if show:
            def _onpick(event):
                ind = np.array(event.ind)[0]
                print('\nTime: %s, Rate: %.3f' %
                      (result[ind, 0].strftime('%Y-%m-%d %H:%m:%S'), result[ind, 1]))
                
            fig, ax = plt.subplots()
            ax.plot(model.columns, model.loc['Ave'], '-',
                      color = '#0072B2', label = 'Average')
            ax.fill_between(model.columns, model.loc['Max'], model.loc['Min'], 
                            color = '#87CEEB', label = 'Confidence Inerval')
            
            for k in np.arange(3):
                m = result[[abv, bel, nor][k]]
                l = ['Above', 'Below', 'Normal']
                col = ['r', 'pink', 'k']
                ax.scatter(m[:, 0], m[:, 1], color = col[k], label = l[k],
                           picker = True)
            
            ax.set_title('Anomaly Detection with built-in model.')
            ax.legend().draggable()       
            fig.canvas.mpl_connect('pick_event', _onpick)             
            ax.grid()
            plt.show()
        
        return [abv, bel, nor]
        
        
        
        
        




