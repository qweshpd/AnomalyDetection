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

'''
   Required library: numpy, pandas, statsmodels.
   Valid only for one-dimension-time-series data. 
'''

class AutoARIAM(object):
    '''
    Compute parameters p,q,dautomatically based on data features.
    
    Attributes:
    set_para: Set parameters.
    get_para: Get parameters.
    test_stat: Stationary test.
    get_d:
    get_pq:
    fit: Fit ARIMA model onto data.
    
    Parameters
    ----------
    conf : number
        Confidence level for statistical test.
    nlag: number
    
    pvalue: number
        
    '''
    def __init__(self):
        self.parameter = {}


def test_stat(timeseries, confid = '1%', show = False,  nlag = 5, pvalue = 0.05):
    '''Stationary Test.'''
    timeseries = np.array(timeseries)
    # Perform Dickey-Fuller test:
    dftest = st.adfuller(timeseries, 1)
    
    if acorr_ljungbox(timeseries, lags = nlag)[1][-1:][0] > pvalue:
        raise ValueError('White noise series!')
        return True
    else:    
        if dftest[0] < dftest[4][confid]:
            return True	
        else:
            print('Not stationary under confidence limits ' + confid + '. Please try a larger one.')
    return False
    
    if show:
        foutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        print('Results of Dickey-Fuller Test:')
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

def getd(timeseries, confid = '1%'):
    '''Get parameter d. '''
    if not confid:
        confid = '1%'
    # If stationary, return d = 0
    if test_stat(timeseries, confid):
        return timeseries, 0
    else:
        tempts = np.diff(timeseries, n = 1)
        try:
            test_stat(tempts, confid)
        except ValueError:
            print('White noise series! Not fit ARIMA')
        else:
            if test_stat(tempts, confid):
                return tempts, 1, timeseries[0]
            else:
                temptts = np.diff(tempts, n = 1)
                if test_stat(temptts, confid):
                    return temptts, 2, tempts[0], timeseries[0]
                else:
                    print('Exceed Differential Limit! Not fit ARIAM!')

def getpq(timeseries, max = 10):
    '''Get parameter p,q based on AIC.'''
    print('Maximum total runtime of 5 minutes are expected. Please be patient and wait your results.')
    order = st.arma_order_select_ic(timeseries, max_ar = max, max_ma = max, ic=['aic', 'bic'])
    return order.aic_min_order

def fitARIMA(times, confid = '1%', p = 0, d = 0, q = 0, max = 10, foreday = 1, show = False):
    '''Fit ARIMA model.'''
    timeseries = np.array(times)
    # timeseries = np.log(np.array(times))
    if p == 0 and q == 0:
        (p, q) = getpq(timeseries, max = max)
    indextemp = getd(timeseries, confid = confid)
    d = indextemp[1]
    tstemp = np.diff(timeseries, d)
    model = ARIMA(timeseries, order = (p, d, q))
    try:
        results_ARIMA = model.fit(disp = -1) 
    except np.linalg.LinAlgError:
        print('SVD did not converge! Please try another p, d, q value.')
        return None
    except ValueError:
        print('Please check your input. You should induce invertibility or choose a different model order.')
        return None
    else:
        #forecast_ARIMA_log = results_ARIMA.forecast(12 * foreday)
        model = _rectsdiff(results_ARIMA.fittedvalues, diffhead = indextemp[2:])
        if show:
            _plotarima(np.array(times), abs(model))
            print('p = %d,'%p + 'q = %d'% q)

        return pd.Series(model, index = times.index) #, forecast_ARIMA_log

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

def _rectsdiff(timesdiff, diffhead = []):
    '''Recover array given difference, index and head data.'''
    l = len(diffhead)
    if l == 0:
        return timesdiff
    elif l == 1:
        return np.insert(np.cumsum(timesdiff), 0, 0) + diffhead[0]
    elif l == 2:
        tsrecover1 = np.insert(np.cumsum(timesdiff), 0, 0) + diffhead[0]
        return np.insert(np.cumsum(tsrecover1), 0, 0) + diffhead[1]

def _plotarima(timeseries, model):
    try:
        import matplotlib.pylab as plt
    except:
        print('matplotlib is not available.')
    else:
        plt.figure()
        plt.subplot(211)
        plt.plot(timeseries, color = 'orange', label = 'Original')
        plt.title('RSS: %.4f'% sum((model - timeseries)** 2))
        plt.legend()
        plt.subplot(212)
        plt.plot(model, color='black', label = 'Model') 
        plt.legend()
        plt.show()



