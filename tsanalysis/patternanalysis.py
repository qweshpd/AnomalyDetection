#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from libs import loadata, process, preproc
import scipy.stats
import time

import warnings
warnings.filterwarnings("ignore")

# matplotlib basic setup
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

#%%  logger setup

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
formatter = logging.Formatter('[%(asctime)-15s] - %(levelname)-8s - %(message)s')

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

fh = logging.FileHandler('PattternAnalysis.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)

#%%
class PatternAnalysis(object):
    '''
    Analyze given device and interface based on a specific pattern type.
    
    Attributes:
        set_para: Set parameters utilized in current algorithm.
        get_para: Get parameters utilized in current algorithm.
        analyze: Analyze a given device.
    '''
    
    def __init__(self):
        self.parameter = {}
        self.startdate = None
        self.endate = None
    
    def set_para(self, paradict):
        '''
        Set parameters utilized in current algorithm.
        
        Parameters
        ----------
        paradict: dict
            Contains parameters to set.
        '''
        for para in paradict:
            self.parameter[para] = paradict[para]
        
    def get_para(self, paralist = None):
        '''
        Get parameters utilized in current algorithm.
        
        Parameters
        ----------
        paralist: list
            Contains parameters to get.
        '''
        if paralist:
            for para in paralist:
                print(para + '\t\t' + str(self.parameter[para]))
        else:
            for para in self.parameter:
                print(para + '\t\t' + str(self.parameter[para]))

    def analyze(self, data):
        '''
        Analyze a given device.
        
        Parameters
        ----------       
        dev_intf: string
            Device name and interface.
        
        Returns
        -------
        output: boolean, list or more
            Algorithm outputs.

        '''
        logger.debug(' ' + self.__class__.__name__)
        timevar = np.array([])
        timevar = np.append(timevar, time.clock())  
        
        logger.debug(' Preprocessing...')
        predata = self._preprocess(data)
        timevar = np.append(timevar, time.clock())
        
        logger.debug(' Processing...')
        prodata = self._process(predata)
        timevar = np.append(timevar, time.clock())
        
        logger.debug(' Outputting...')
        output = self._getoutput(data, prodata)
        timevar = np.append(timevar, time.clock())
        
        timeused = timevar[1:] - timevar[:-1]
        logger.debug(' Total time used:\t%.6f seconds.\n\
                     \t\t\tPreprocess\t\t%.6f seconds\n\
                     \t\t\tProcess\t\t\t%.6f seconds\n\
                     \t\t\tOutput \t\t\t%.6f seconds' 
                     % (timevar[-1] - timevar[0], timeused[0], timeused[1], timeused[2]))
        
        return output
    
    def _preprocess(self, data):
        return data
    
    def _process(self, data):
        raise NotImplementedError
        
    def _getoutput(self, data, pdata):
        raise NotImplementedError
        
    def help(self):
        print(PatternAnalysis.__doc__)
        print('    ' + self.__class__.__name__)
        print(self.__doc__)    
        
#%% local peak analysis
 
class LocalPeakanalysis(PatternAnalysis):
    '''
    Find local peaks within timeseries if any. Parameters below are to be set 
    before analyzing, or their default values will be adopted.
    
    Parameters
    ----------
    mph : {None, number}, optional (default = None)
        Detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        Detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        Detect peaks that are greater than 'threshold' in relation to 
        their immediate neighbors.
    ind : positive integer, optional (default = 10)
        Daximum number of peaks to be shown (in descending order).
    show : bool, optional (default = False)
        If True, plot data in matplotlib figure.
    '''
    
    def __init__(self):
        super().__init__()
        self.parameter['mph'] = 0
        self.parameter['mpd'] = 1
        self.parameter['thre'] = 0
        self.parameter['ind'] = 10
        self.parameter['show'] = False
    
    def _process(self, data):
        '''
        Find peaks in data based on their amplitude and other features.
        '''
            
        index = process.detect_peaks(data, mph = self.parameter['mph'],
                                mpd = self.parameter['mpd'],
                                threshold = self.parameter['thre'])
        return index
    
    
    def _getoutput(self, data, index):
        value = []
        
        for i in index:
            value.append(data[i])
        
        num = min(self.parameter['ind'], len(index))
        ind = np.argsort(value)[::-1][:num]

        for timestamp in ind:
                print(str(data.index[index[timestamp]]) + ' --> ' + str(data[index[timestamp]]))
        
        if self.parameter['show']:
            plt.figure()
            plt.plot(data)
            for timestamp in ind:
                plt.scatter(data.index[index[timestamp]], 
                            data[index[timestamp]], color = 'red')
        
        return index

#%% cycle analysis
 
class CycleAnalysis(PatternAnalysis):
    '''
    Find cycles within timeseries if any. Parameters below are to be set 
    before analyzing, or their default values will be adopted.
    
    Parameters
    ----------
    mc : positive integer, optional (default = np.inf)
        Maximum cycle to be detected
    show : bool, optional (default = False)
        If True, plot data in matplotlib figure.
    '''
    
    def __init__(self):
        super().__init__()
        self.parameter['mph'] = 0
        self.parameter['mpd'] = 1
        self.parameter['thre'] = 5
        self.parameter['mc'] = np.inf
        self.parameter['show'] = False
        
    def _process(self, data):
        
        num = math.ceil(len(data)/ 2)
        fftf = np.fft.fft(data) * 2 / len(data)
        
        if not self.parameter['mph']:
            self.parameter['mph'] = np.mean(data) / 3
            
        frequency = process.detect_peaks(abs(fftf)[0: num],
                        mph = self.parameter['mph'],
                        mpd = self.parameter['mpd'],
                        threshold = self.parameter['thre']) 
        
        if self.parameter['show']:
            plt.figure()
            plt.plot(abs(fftf)[0: num])
            for i in frequency:
                plt.scatter(i, abs(fftf)[0: num][i], color = 'r')
            plt.show()
            
        return [i for i in frequency/12 if i < self.parameter['mc']]
                                    
    def _getoutput(self, data, frequency):
        if len(frequency):
            print(' days, '.join('%.3f' % i for i in frequency) + ' days\n')
        else:
            print('No cycle found with the given conditions.\n')
            

#%% similarity analysis   

class SimilarityAnalysis(PatternAnalysis):
    '''
    Analyze similarity between two interfaces. Parameters below are to be set 
    before analyzing, or their default values will be adopted.
    
    Parameters
    ----------
    thre : positive number, optional (default = 0.5)
        Threshold of similarity if simply determine whether similar
    slot : positive number, optional (default = 80)
        Time range of a possible similar part        
    show : bool, optional (default = False)
        If True, plot data in matplotlib figure.
    '''
    
    def __init__(self):
        super().__init__()
        self.parameter['thre'] = 0.5
#        self.parameter['amp'] = None
        self.parameter['slot'] = 80
        self.parameter['show'] = True
        self.threshold = 0.85
        self.drift = 0.01
    
    def _preprocess(self, ts):
        if not len(ts[0]) == len(ts[1]):
            print('Two data series must be of the same length!')
            raise
            
        if not self.parameter['slot']: # care only whether similar
            tseries = []
            for data in ts:
                tseries.append(preproc.normalize(data, '-11'))
            return tseries
        else: # care whole parts
            return ts
    
    def _process(self, predata):       
        score = scipy.stats.pearsonr(predata[0], predata[1])[0]
        
        if abs(score) < self.parameter['thre']:
            return []
        else: # similar
            if not self.parameter['slot']: # only care similarity
                return [1]
            else: # simliarity and slot
                k = np.array(predata[0]) - np.array(predata[1])
                temp = (k - np.min(k)) / (np.max(k) - np.min(k))
                indexes = process.detect_cusum(temp, self.threshold * np.std(temp), 
                                          self.drift, 1)    
            return [indexes[1], indexes[2]]
        
    def _getoutput(self, data, prodata):  
        if not prodata:
            print('Two time series of no similarity!')
        else:
            print('Two similar time series data!')
            if len(prodata) == 2:
                ind = []
                tai = prodata[0]
                taf = prodata[1]
                for i in np.arange(1, len(tai)):
                    if tai[i] - taf[i - 1] >= self.parameter['slot']:
                        ind.append([taf[i - 1], tai[i]])
                        print(data.index)
                        print(tai[i])

                if self.parameter['show']:
                    self.__plot(data, ind)
                    
    
    def __plot(self, data, index):
        plt.figure()
        plt.plot(data[0], 'b')
        plt.plot(data[1], 'y')
        plt.plot([], [], 'r', label = 'Similar Part')
        for sim in index:
            plt.axvline(x = sim[0], color = 'red')
            plt.axvline(x = sim[1] - 1, color = 'red')
            plt.plot(np.linspace(sim[0], sim[1] - 1, sim[1] - sim[0]), 
                     data[0][sim[0]:sim[1]], 'r')
            plt.plot(np.linspace(sim[0], sim[1] - 1, sim[1] - sim[0]), 
                     data[1][sim[0]:sim[1]], 'r')
        plt.legend()
        plt.show()
#%% on off analysis

class OnsetAnalysis(PatternAnalysis):
    '''
    Analyze onset in data based on amplitude threshold.
    
    One of the simplest methods to automatically detect the data onset is 
    based on amplitude threshold, where the signal is considered to be 'on' 
    when it is above a certain threshold. The function implements such onset 
    detection based on the amplitude-threshold method. Note: You might have to
    tune the parameters according to the signal-to-noise characteristic
    of the data.

    Parameters
    ----------
    thre : number, optional (default = 0)
        Minimum amplitude of to detect.
    nab : number, optional (default = 1)
        Minimum number of continuous samples >= 'thre' to detect.
    nbe : number, optional (default = 0)
        Minimum number of continuous samples below 'thre' that will be ignored 
        in the detection of 'x' >= 'thre'.
    thre2 : number or None, optional (default = None)
        Minimum amplitude of 'nab2' values in 'x' to detect.
    nab2 : number, optional (default = 1)
        Minimum number of samples >= 'thre2' to detect.
    show : bool, optional (default = False)
        If True, plot data in matplotlib figure.
    '''
    
    def __init__(self):
        super().__init__()
        self.parameter['thre'] = 0
        self.parameter['nab'] = 1
        self.parameter['nbe'] = 0 
        self.parameter['thre2'] = None
        self.parameter['nab2'] = 1
        self.parameter['show'] = True

    def _process(self, data):
        '''
        Detects onset in data based on amplitude threshold.
        '''

        inds = process.detect_onset(data, threshold = self.parameter['thre'], 
                                  n_above = self.parameter['nab'], 
                                  n_below = self.parameter['nbe'],
                                  threshold2 = self.parameter['thre2'], 
                                  n_above2 = self.parameter['nab2'])
        return inds
    
    def _getoutput(self, data, inds):
        print('\n%d times of onset detected.\n' % len(inds))
        print('Initial Time \t\t\tFinal Time')
        for i in inds:
            print(str(data.index[i[0]]) + '\t' + str(data.index[i[1]]))
            
        if self.parameter['show']:
            self._plot(data, inds)
        

    def _plot(self, data, inds):
        '''
        Plot results of the detect_onset function.
        '''    
        
        _, ax = plt.subplots(1, 1, figsize = (15, 6))

        if inds.size:
            for (indi, indf) in inds:
                if indi == indf:
                    ax.plot(indf, data[indf], 'ro', mec = 'r', ms = 6)
                else:
                    ax.plot(range(indi, indf + 1), data[indi:indf + 1], 'r', lw = 1)
                    ax.axvline(x = indi, color = 'b', lw = 1, ls='--')
                ax.axvline(x = indf, color = 'b', lw = 1, ls = '--')
            inds = np.vstack((np.hstack((0, inds[:, 1])),
                              np.hstack((inds[:, 0], data.size - 1)))).T
            for (indi, indf) in inds:
                ax.plot(range(indi, indf + 1), data[indi:indf + 1], 'k', lw = 1)
        else:
            ax.plot(data, 'k', lw = 1)
            ax.axhline(y = self.parameter['thre'], color = 'r', lw = 1, ls='-')

        plt.show()

#%% status change  analysis
        
class StatusAnalysis(PatternAnalysis): 
    '''
    Analyze status (enumerated type) in a given pattern. Parameters below are 
    to be set before analyzing, or their default values will be adopted. 
    
    Note： More functions are under development.
    
    Parameters
    ----------
    status: set, optional (default = set())
        Given set of known statuses.
    scan : bool, ooptional (default = False)
        If True, scan and return the index where status not in 'given status'.
    show : bool, optional (default = True)
        If True, plot data in matplotlib figure.
    '''
    
    def __init__(self):
        super().__init__()
        self.parameter['status'] = set()
        self.parameter['scan'] = False
        self.parameter['show'] = True

    def _preprocess(self, data):
        if not self.parameter['status']:
            self.parameter['current status'] = set(data)
            
        return data
            
    
    def _process(self, data):
        prodata = []
        for i in np.arange(1, len(data)) :
            if not data[i] == data[i - 1]:
                prodata.append(i)
        return prodata
    
    def _getoutput(self, data, index):
        
        print('The data has %d statuses: \n' % len(set(data)) + str(sorted(set(data))))
        print('\nIt has changed %d times in total.\n' % len(index))
        
        for i in index:
            print(str(data.index[i]) + '\t' + str(data[i - 1]) +
                  '\t ==> \t' + str(data[i]))
        
        if self.parameter['show']:
            self._plot(data, index)

        
        del self.parameter['current status']
        return index
    
    def _plot(self, data, index):
        plt.figure()
        plt.plot(data, color = 'blue', label = 'data')
        plt.plot([], [], color = 'red', label = 'change')
        for i in index:
            plt.plot([i - 1, i], data[i - 1 : i + 1], 'r-', 
                     marker = 'o')
        plt.legend()
        plt.show()
            
#%% trend analysis

class TrendAnalysis(PatternAnalysis):   
    '''
    Analyze trend of given time series data. 
    
    Change detection refers to procedures to identify abrupt changes in 
    a phenomenon. By abrupt change it is meant any difference in relation to 
    previous known data faster than expected of some characteristic of 
    the data such as amplitude, mean, variance, frequency, etc.
    
    Parameters below are to be set before analyzing, or their default values 
    will be adopted.
    
    Parameters
    ----------
    thre : positive number, optional (default = 1)
        Amplitude threshold for the change in the data
    drift : positive number, optional (default = 0)
        Drift term that prevents any change in the absence of change
    ending : bool, optional (default = True)
        True to estimate when the change ends; False otherwise.
    show : bool, optional (default = False)
        If True, plot data in matplotlib figure.
    '''
    def __init__(self):
        super().__init__()
        self.parameter['thre'] = 1
        self.parameter['drift'] = 0
        self.parameter['end'] = True
        self.parameter['show'] = True

    def _process(self, data):

            
        ta, tai, taf, amp, gp, gn = process.detect_cusum(data, self.parameter['thre'],
                                    self.parameter['drift'], 
                                    self.parameter['end'])
        
        return [ta, tai, taf, amp, gp, gn]
    
    def _getoutput(self, data, index):
        
        ta, tai, taf, amp, gp, gn = index
 
        if self.parameter['show']:
            process._plotcusum(data, self.parameter['thre'], self.parameter['drift'],
                     self.parameter['end'], ta, tai, taf, gp, gn)
        return [ta, tai, taf, amp]

#%% saturation analysis
    
class SaturationAnalysis(PatternAnalysis):  
    '''
    Analyze saturation starus of given time series data. 
    
    Saturation status refers to a traffic where its value keeps stable to a
    fixed range between a minimum and maximum value. Generally speaking, most
    saturation can be regarded as an onset which threshold is, say, 95% of the 
    max value.
    
    Parameters below are to be set before analyzing, or their default values 
    will be adopted.
    
    Parameters
    ----------
    thre : positive number, optional (default = '90%')
        Amplitude threshold for the change in the data
    nab : number, optional (default = 20)
        Minimum number of continuous samples to detect.
    nbe : number, optional (default = 5)
        Minimum number of continuous samples below 'thre' that will be ignored.
    show : bool, optional (default = False)
        If True, plot data in matplotlib figure.
    '''    
    
    def __init__(self):
        super().__init__()
        self.parameter['thre'] = '90%'
        self.parameter['nab'] = 20
        self.parameter['nbe'] = 5 
        self.parameter['show'] = True
    
    def _process(self, data):
        
        if type(self.parameter['thre']) == type(''):
            thres = max(data) * float(self.parameter['thre'].split('%')[0])/100
        elif type(self.parameter['thre']) == type(0):
            thres = self.parameter['thre']
        elif type(self.parameter['thre']) == type(0.0):
            thres = self.parameter['thre']
        
        inds = process.detect_onset(data, threshold = thres, 
                                  n_above = self.parameter['nab'], 
                                  n_below = self.parameter['nbe'])
        return inds
    
    def _getoutput(self, data, inds):
        
        print('Initial Time \t\t\tFinal Time')
        for i in inds:
            print(str(data.index[i[0]]) + '\t' + str(data.index[i[1]]))
            
        if self.parameter['show']:
            plt.figure()
            plt.plot(data)
            for (indi, indf) in inds:
                plt.plot(data.index[indi:indf], data[indi:indf], 'r-')
                
        return inds


#%% no data analysis
        
class NodataAnalysis(PatternAnalysis):   
    '''
    Analyze and provide solution to missing data. Parameters below are to be 
    set before analyzing, or their default values will be adopted.
    
    Parameters
    ----------
    offset : A number of string aliases are 
        Given to useful common time series frequencies.
    '''

    def __init__(self):
        super().__init__()
        self.parameter['offset'] = '2H'
        self._timesheet = {'S':1, 'M':60, 'H':3600, 'D':24*3600}
        
    def _process(self, data):        
        slot = int(self.parameter['offset'][:-1]) * \
                    self._timesheet[self.parameter['offset'][-1]] # convert parameter to seconds
        pdata = data.dropna() # drop NaN data
        periods = pdata.index[1:] - pdata.index[:-1] # time period in real data
        temp = np.where(periods.total_seconds() > slot)
        return temp + 1
        
    def _getoutput(self, data, index):
        if not len(index):
            print('No missing data in current data flow.')
        else:
            print('Unexpected error happended %d times.' % len(index))
            print('Initial Time \t\t\tFinal Time')
            for ind in index:
                print(data.index[ind - 1].strftime('%Y-%m-%d %H:%M:%S') \
                      + '\t' + data.index[ind].strftime('%Y-%m-%d %H:%M:%S') )
