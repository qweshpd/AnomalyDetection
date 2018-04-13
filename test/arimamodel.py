#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARIMA

import matplotlib.pylab as plt
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 15, 6

import warnings
warnings.filterwarnings("ignore")

class AutoARIAM(object):
    '''
    Compute parameters p,q,dautomatically based on data features.
    Required library: numpy, pandas, statsmodels.
    Valid only for one-dimension-time-series data. 
    
    Attributes
    ----------
    set_para: Set parameters.
    get_para: Get parameters.
    test_stat: Stationary test.
    get_d: Get proper d value for stational timeseries.
    get_pq: Get p,q values to fit ARIMA model.
    fit: Fit ARIMA model onto data.
    '''
    
    def __init__(self):
        self.valid = False

    def _valid_test(self, timeseries, nlag = 5, conf = '1%', pvalue = 0.05, 
                  show = True):
        '''
        Test whether time series data fit for ARIMA model. Ljungbox test and
        Dickey-Fuller test included.
        
        Parameters
        ----------
        timeseries : 1-D pandas Series object or numpy array
            The time-series to which to fit the ARIMA estimator.
        nlag : integer 
            The largest lag to be considered for Ljungbox test.  
        conf : number
            Confidence level for statistical test.      
        pvalue : float
            Threshold to which test result is compared.
        show : bool, optional (default = True)
            If True, print Dickey-Fuller test result.
        '''
        
        self.nlag = nlag
        self.conf = conf
        self.pvalue = pvalue
        
        if not conf in ['1%', '5%', '10%']:
            raise KeyError('Please input a valid confidence level!')
            
        if acorr_ljungbox(timeseries, lags = nlag)[1][-1:][0] > self.pvalue:
            raise ValueError('White noise series!')
            
        ts = np.array(timeseries)                 
        dftest = st.adfuller(ts, 1) # Perform Dickey-Fuller test

        dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic',
                    'p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
            
        if show:
            print('Results of Dickey-Fuller Test:')
            print(dfoutput)
        
        if dftest[0] < dftest[4][self.conf]:
            self.valid = True
            return dfoutput, True
        else:
            print('Not stationary under confidence limits ' + self.conf + 
                  '. Please try a larger one.')
            return dfoutput, False

    def _get_d(self, timeseries, nlag = 5, conf = '1%', pvalue = 0.05, 
                  show = True):
        '''
        Calculate parameter d for difference order.
        
        Parameters
        ----------
        timeseries : 1-D pandas Series object or numpy array
            The time-series to which to fit the ARIMA estimator.
        nlag : integer 
            The largest lag to be considered for Ljungbox test.  
        conf : number
            Confidence level for statistical test.      
        pvalue : float
            Threshold to which test result is compared.
        show : bool, optional (default = True)
            If True, print Dickey-Fuller test result.
        '''
        res = self._valid_test(timeseries, nlag = nlag, conf = conf, 
                            pvalue = pvalue, show = False)
        if res[1]:    # if stationary, return d = 0
            self.validts = timeseries
            self.d = 0 
        else:    # if not stationary, try difference once
            tempts = np.diff(timeseries, n = 1)
            res = self._valid_test(tempts, nlag = nlag, conf = conf, 
                            pvalue = pvalue, show = False)
            if res[1]:
                self.validts =  tempts
                self.d = 1
                self.head = timeseries[0]
            else:
                tempts2 = np.diff(timeseries, n = 2)
                res = self._valid_test(tempts2, nlag = nlag, conf = conf, 
                                        pvalue = pvalue, show = False)
                if res[1]:
                    self.validts = tempts2
                    self.d = 2
                    self.head = [tempts[0], timeseries[0]]
                else:
                    raise ValueError('Exceed Dfferential Limit! Not fit for ARIMA!')
        
        if show:
            # Determing rolling statistics
            ts = pd.Series(self.validts)
            rolmean = pd.rolling_mean(ts, window = 12)
            rolstd = pd.rolling_std(ts, window = 12)
            
            # Plot rolling statistics:
            plt.figure()
            plt.plot(ts, 'b', label = 'Original')
            plt.plot(rolmean, 'r', label = 'Rolling Mean')
            plt.plot(rolstd, 'k', label = 'Rolling Std')
            plt.legend(loc = 'best')
            plt.title('Rolling Mean & Standard Deviation After %d Difference.' 
                      % self.d)
            plt.show(block = False)
            print('Results of Dickey-Fuller Test:')
            print(res[0])

        return self.validts
    
    def _get_pq(self, timeseries, max_ar = 10, max_ma = 10, ic = 'aic'):
        '''
        Calculate parameter p,q for ARIMA model based on AIC or BIC order.
        
        Parameters
        ----------
        timeseries : 1-D pandas Series object or numpy array
            The time-series to which to fit the ARIMA estimator.
        max_q : integer 
            Maximum number of q to fit.
        max_p : integer 
            Maximum number of p to fit.
        ic : string
            Information criteria to report.
        '''
        
        print('Maximum total runtime of 5 minutes are expected.')
        print('Calculating... Please be patient and wait your results.')
        order = st.arma_order_select_ic(timeseries, max_ar = max_ar, 
                                        max_ma = max_ma, ic = ic)
        if ic == 'aic':
            self.p, self.q = order.aic_min_order
        elif ic == 'bic':
            self.p, self.q = order.bic_min_order
        
        return self.p, self.q

    def fit(self, tsdata, max_ar = 10, max_ma = 10, ic = 'aic', 
            nlag = 5, conf = '1%', pvalue = 0.05, show = False, **kwargs):
        '''
        Calculate parameter p,q for ARIMA model based on AIC or BIC order.
        
        Parameters
        ----------
        tsdata : 1-D pandas Series object or numpy array
            The time-series to which to fit the ARIMA estimator.
        max_q : integer 
            Maximum number of q to fit.
        max_p : integer 
            Maximum number of p to fit.
        ic : string
            Information criteria to report.
        nlag : integer 
            The largest lag to be considered for Ljungbox test.  
        conf : number
            Confidence level for statistical test.      
        pvalue : float
            Threshold to which test result is compared.
        show : bool, optional (default = True)
            If True, print Dickey-Fuller test result.
        kwargs : number
            Pre-given p,d,q values, foredays to forcast.
        '''
        timeseries = np.array(tsdata)     
        
        if 'd' in kwargs.keys():
            self.d = kwargs['d']
            if self.d == 1:
                self.head = timeseries[0]
            elif self.d == 2:
                self.head = [timeseries[1] - timeseries[0], timeseries[0]]
        else:
            _ = self._get_d(timeseries, nlag = nlag, conf = conf, 
                            pvalue = pvalue, show = show)
            
        if ('p' in kwargs.keys()) and ('q' in kwargs.keys()):
            self.p = kwargs['p']
            self.q = kwargs['q']
        else:
            _, _ = self._get_pq(timeseries, max_ar = max_ar, max_ma = max_ma, 
                                ic = ic)
        model = ARIMA(self.validts, order = (self.p, self.d, self.q))
        
        try:
            results_ARIMA = model.fit(disp = -1) 
        except np.linalg.LinAlgError:
            print('SVD did not converge! Please try another p, d, q value.')
        except ValueError:
            print('Please check your input.')
            print('You should induce invertibility or choose a different model order.')
        else:
            self.fitmodel = self._rectsdiff(results_ARIMA.fittedvalues, 
                                       diffhead = self.head)
            if show:
                self._plotarima(np.array(timeseries), abs(self.fitmodel))
                print('p = %d, q = %d'% (self.p, self.q))

        if 'fore' in kwargs.keys():
            self.forecast = results_ARIMA.forecast(kwargs['fore'])
            return pd.Series(self.fitmodel, index = tsdata.index), self.forecast

        return pd.Series(self.fitmodel, index = tsdata.index)

def getalarm(timesa, timestamp, p = 0, d = 0, q = 0, max = 10, show = False):
    '''Print alarm timestamps'''
    mean = np.mean(timesa)
    ts_model = abs(fitARIMA(timesa, p = p, q = q, show = show))
    anomal = np.where((timesa > 1.5 * ts_model) & (timesa > mean))[0]
    # anomal = np.where(ts > 2 * pd.Series(ts_model).shift(periods = -1))[0]
    print('Time')
    for i in anomal:
        print(timestamp[i])
    if show:
        plt.figure()
        plt.plot(timesa, color = 'orange', label = 'Original')
        plt.plot(ts_model, color='black', label = 'Model')
        # plt.scatter(anomal, [timesa[i] for i in anomal], color = 'red', label = 'Anomal')
        plt.legend()
        plt.show()

    def _rectsdiff(self, timesdiff, diffhead = []):
        '''
        Recover array given difference, index and head data.
        
        Parameters
        ----------
        timesediff : 1-D pandas Series object or numpy array
            The differenced time-series data.
        diffhead : number, list
            List of head data before differenced.
        '''

        l = len(diffhead)
        if l == 0:
            timeseries = timesdiff
        elif l == 1:
            timeseries = np.insert(np.cumsum(timesdiff), 0, 0) + diffhead[0]
        elif l == 2:
            tsrecover1 = np.insert(np.cumsum(timesdiff), 0, 0) + diffhead[0]
            timeseries = np.insert(np.cumsum(tsrecover1), 0, 0) + diffhead[1]
        
        return timeseries
    
    def _plotarima(self, timeseries):
        '''
        Plot fitted ARIMA model versus original data.
        
        Parameters
        ----------
        timesediff : 1-D pandas Series object or numpy array
            The differenced time-series data.
        '''

        plt.figure()
        plt.subplot(211)
        plt.plot(timeseries, 'o', label = 'Original')
        plt.title('RSS: %.4f'% sum((self.model - timeseries)** 2))
        plt.legend()
        plt.subplot(212)
        plt.plot(self.model, 'k', label = 'Model') 
        plt.legend()
        plt.show()



