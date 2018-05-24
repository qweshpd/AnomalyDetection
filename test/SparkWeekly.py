#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import datetime

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
    def __init__(self, holiday = [], freq = 2, filt = None, sc = None):
        '''
        Initialize the model with holidays.
        
        Parameters
        ----------       
        holiday: list of string, YYYY-MM-DD
            Predefined holidays.
        sc: pyspark.SparkContext
            The SparkContext, required.
        freq: int
            The expected data frequency.
        filt: function
            The function of how data model will be build.
        '''
        assert sc != None, "Missing SparkContext"
        
        if filt is None:
            filt = lambda x: [np.array(x).mean(), np.array(x).mean() + 3 * np.array(x).std(),\
                              np.array(x).mean() - 3 * np.array(x).std()]
        self.filt = filt
        self.sc = sc
        self.holiday = holiday
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
        data : pandas.Series data with datetime-like index
            Data to be analyzed.
        '''
        sc = self.sc
        holiday = self.holiday
        data = self.data
        freq = self.freq
        tmp_data = np.vstack((list(data.index), list(data))).T
        
#        hol_day = self.sc.parallelize(tmp_data).filter(lambda x: x[0].strftime('%Y-%m-%d') in self.holiday)
        reg_day = sc.parallelize(tmp_data).filter(lambda x: x[0].strftime('%Y-%m-%d') not in holiday)        
        off_day = sc.parallelize(tmp_data).filter(lambda x: (x[0].strftime('%Y-%m-%d') in holiday) or (x[0].weekday() > 4))

        reg_day_data = reg_day\
                  .map(lambda x:(str(x[0].weekday()) + ('0' + str(int(x[0].hour/freq)))[-2:], [x[1], x[0]]))\
                  .reduceByKey(lambda x,y:np.vstack((np.array(x),np.array(y)))).sortByKey()
                  
        off_day_data = off_day\
                  .map(lambda x:(('0' + str(int(x[0].hour/freq)))[-2:], [x[1], x[0]]))\
                  .reduceByKey(lambda x,y:np.vstack((np.array(x),np.array(y)))).sortByKey()
        
        for i in np.arange(7):        
            tmp_data = reg_day_data.filter(lambda x: x[0].startswith(str(i)))\
                 .map(lambda x:(x[0][1:], x[1]))
            self.dailydata[_eachday[i]] = tmp_data.collectAsMap()                 

        self.dailydata['Offday'] = off_day_data.map(lambda x:(x[0][1:], x[1])).collectAsMap()
  
        return reg_day_data, off_day_data
        
    def fit(self, data, filt = None):

        filt = self.filt
            
        self.data = data
        reg_day_data, off_day_data = self._extract_day()
        
        bus = pd.DataFrame([])
        col = []
        for i in np.arange(7):
            rdd = reg_day_data.filter(lambda x:x[0].startswith(str(i)))\
                                      .map(lambda x:(x[0][1:], x[1][:, 0]))
            tmpmodel = self._build_model(rdd, filt)                         
            col.append(_eachday[i][:3] + k for k in self.columns)
            bus = bus.T.append(tmpmodel.T).T
            self.dailymodel[_eachday[i]] = tmpmodel
#        bus.columns = col
        
        offrdd = off_day_data.map(lambda x:(x[0][1:], x[1][:, 0]))
        
        self.dailymodel['Busday'] = bus
        self.dailymodel['Offday'] = self._build_model(offrdd, filt) 

        
    def _build_model(self, RDDdata, filt):
        '''
        Build daily model based on data.  
               
        Parameters
        ----------       
        RDDday : spark.RDD object
            Day to build daily model.
        '''
        col = self.columns
        model = RDDdata.map(lambda x: (x[0], filt(x[1]))).collectAsMap()
        modeldf = pd.DataFrame(model, columns=col, index=_index)
        
        return modeldf
        
#%%%       
            
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
        
        
        
        
        




