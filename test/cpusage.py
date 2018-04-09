#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''A test script for online anomaly detection based on Kalman Filter.'''

import psutil
import matplotlib.pyplot as plt
import time  
import warnings
warnings.filterwarnings("ignore")

seconds = 30 # total runtime

plt.close('all')
plt.ion()
psutil.cpu_percent(percpu = True)
logicnum = psutil.cpu_count(logical = True)
times = []
for i in range(logicnum):
    vars()['CPU' + str(i)] = [psutil.cpu_percent(percpu = True)[i]]
   
for i in range(seconds):
    usage = psutil.cpu_percent(percpu = True)
    times.append(time.strftime("%H:%M:%S", time.localtime()))
    for k in range(logicnum):
        vars()['CPU' + str(k)].append(usage[k])
        plt.subplot(2, int(logicnum / 2), k + 1)
        if usage[k] > 60.0 and i > 3:  
            print('\nLogical CPU' + str(k) + ' Usage: %.2f'% usage[k])
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            plt.plot([i - 1, i], vars()['CPU' + str(k)][-2:], 'r-')
        else:
            plt.plot([i - 1, i], vars()['CPU' + str(k)][-2:], 'b--')
        plt.title('Logical CPU' + str(k))
        plt.ylim([0, 100])
    plt.pause(1)

plt.show()

  
