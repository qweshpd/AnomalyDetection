# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import scipy.stats
import pattern.patternanalysis as pa
import libs.preproc as pp

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

import warnings
warnings.filterwarnings("ignore")

#%%
ltc = pd.read_csv('.\data\LTC.csv')['Close Price']
ltc.dropna(inplace = True)
eth = pd.read_csv('.\data\ETH.csv')['Close Price']
bch = pd.read_csv('.\data\BCH.csv')['Close Price']
#%%
kk = pp.normalize(bch - eth, '01')
plt.figure()
plt.plot(kk)
plt.figure()
plt.boxplot(kk)
plt.figure()
plt.hist(kk)
plt.figure()
plt.plot(bch, 'r')
plt.figure()
plt.plot(eth, 'b')
#%%
ta2 = pa.TrendAnalysis()
ta2.set_para({'thre':np.std(kk)*0.9, 'drift':0})
indexes = ta2.analyze(kk)