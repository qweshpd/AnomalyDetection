# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm
#from itertools import groupby

_scale = 10
_threshold = 60

class auto_onehot(object):
    '''
    Encode categorical integer features using one-hot.
    
    Parameters
    --------
    featuredic : dict
        All features with feature name.    
    
    Examples
    --------
    
    >>>feature = {'fe1': ['ab', 'bc', 'ca'],
                  'fe2': [10*i for i in np.arange(8)],
                  'fe3': ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}
    >>>auto_onehot(feature).transform(['ab', 60, 'Sun'])
    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.])
    '''
    
    def __init__(self, featuredic):
        self.feature = list(featuredic.keys())
        self.fnum = len(featuredic.keys())
        self.attrnum = []
        for fea in featuredic.keys():
            setattr(self, fea, featuredic[fea])
            self.attrnum.append(len(featuredic[fea]))
        self.vnum = sum(self.attrnum)    
    
    def transform(self, slicefeature):
        '''
        Fit OneHotEncoder to X, then transform X.
        
        Parameters
        ----------
        X : array-like, list of feature instances.
            Input array of features.
        '''
        assert len(slicefeature) == self.fnum, 'please input only %d feature(s)'%self.fnum
        feature_array = np.zeros((1, self.vnum))[0]
        for i in np.arange(self.fnum):
            allvalue = getattr(self, self.feature[i])
            tmpnum = sum(self.attrnum[:i]) + allvalue.index(slicefeature[i])
            feature_array[tmpnum] = 1
        return feature_array


class NBEnum(object):
    '''
    Automatical anomaly detection for enum-like variables.
    '''
    
    def __init__(self, var_name, datalist):
        self.name = var_name
        self.data = datalist
        self.features = list(set(datalist))
        self.encode = auto_onehot({self.name: self.features})
        self.model = pd.DataFrame(columns=self.features, 
                                  index = ['mu', 'sigma', 'tmax', 'tmin'])
        
    def _encoding(self, all_data):
        '''Encode categorical integer features using one-hot.'''
        tranf = self.encode.transform
        code_array = np.vstack(list(map(lambda x:tranf([x]), all_data)))
        self.code_array = code_array
        
        return code_array
    
    def _seq_analy(self, code, value = 1):
        '''
        Analysis sequential code.
        
        Parameters
        ----------
        code : array-like, contains only 0 or 1
            Code data to be analyzed.
        value : int, 0 or 1
            Target value.
        '''
        isvalue = np.concatenate(([0], np.equal(code, value), [0]))
        inds = np.where(np.abs(np.diff(isvalue)) == 1)[0].reshape(-1, 2)
        last = inds[:, 1] - inds[:, 0]
        inds[:, 1] = inds[:, 1] - 1
        return np.vstack((inds[:, 0], inds[:, 1], last)).T
    
    def _freq_anal(self, nlist):
        '''Statistical matrix.'''
        dmean = nlist.mean()
        dstd = nlist.std()
        factor = np.sqrt(-np.log(1 - _threshold/100)/_scale)
        dmax = dmean + dstd * factor
        dmin = max(0, dmean - dstd * factor)

        return  dmean, dstd, dmax, dmin
    
    def modeling(self):
        '''
        Build statistical model based on historical data.
        '''
        features = self.features
        code_array = self._encoding(self.data)
        self.feature_array = {}
        for i in np.arange(self.encode.attrnum[0]):
            tmparray = self._seq_analy(code_array[:, i])
            self.feature_array[features[i]] = tmparray
            self.model[features[i]] = self._freq_anal(tmparray[:, 2])
            
    def analyze(self, data = None, show = False):
        '''
        Analyze data based on model.
        
        Parameters
        ----------
        data : 
            Data to be analyzed.
        show : bool, optional (default = False)
            Show plot (True) or not (False).
        
        Returns
        -------
        inds : 2D array-like [indi, indf]
            Initial and final indices of data detected as anomaly.
        '''
        features = self.features
        if data is None:
            tmpdict = self.feature_array
        else:
            tmpdict = {}
            encode_data = self._encoding(data)
            for i in np.arange(self.encode.attrnum[0]):
                tmparray = self._seq_analy(encode_data[:, i])
                tmpdict[features[i]] = tmparray
        
        model = self.model
        alert = []
        for onef in features:
            model_array = np.array(model[onef])
            test_array = tmpdict[onef][:, 2]
            inds = np.where(np.logical_or(test_array > model_array[2],
                                          test_array < model_array[3]))[0]
            
            if inds.any():
                alert.append([onef, tmpdict[onef][inds][:, :2]])
        
        if show:
            try:
                import matplotlib.pylab as plt
                plt.rcParams['figure.figsize'] = 15, 6
            except ImportError:
                print('matplotlib is not available.')
            else:
                plt.figure()
                for fi in features:
                    entry = features.index(fi)
                    pltmodel = np.array(model[fi])
                    test_array = tmpdict[fi][:, 2]
                    
                    plt.subplot(len(features), 1, entry + 1)
                    
                    base = np.arange(-10, max(test_array)+20, 1)
                    normal = norm.pdf(base, pltmodel[0], pltmodel[1])
                    anomalous = np.logical_or(base > pltmodel[2],
                                              base < pltmodel[3])
                    
                    plt.hist(test_array, bins=base-0.5, 
                             normed=True, zorder=1)
                    plt.fill_between(base, normal, where=anomalous, 
                                     color = [1,0,0,0.4], zorder = 2)
                    plt.plot(base, normal, color = 'black', zorder = 3)
                    plt.title('Feature distribution of %s'%fi)
                    plt.show()
        
#        if np.shape(alert)[0]:
#            alert = np.sort(np.vstack(alert), axis = 0)
            
        return alert
    