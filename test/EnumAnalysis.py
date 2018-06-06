# -*- coding: utf-8 -*-

import numpy as np
#from itertools import groupby

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
        
    def _encoding(self):
        tranf = self.encode.transform
        all_data = self.data
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
        isvalue = np.concatenate(([0], np.equal(li, value), [0]))
        inds = np.where(np.abs(np.diff(isvalue)) == 1)[0].reshape(-1, 2)
        last = inds[:, 1] - inds[:, 0]
        inds[:, 1] = inds[:, 1] - 1
        return np.vstack((inds[:, 0], inds[:, 1], last))
    
    def _freq_anal(self, nlist):
        num, freq = np.histogram(nlist, bins=np.arange(1, max(nlist)+2) - 0.5)
        return np.vstack((num, (freq + 0.5)[:-1]))
    
    def modeling(self):
        code_array = self._encoding()
        ind = {}
        for i in np.arange(self.encode.attrnum[0]):
            tmparray = self._seq_analy(code_array[:, i])
            ind[self.features[i]] = self._freq_anal(tmparray[:, 3])
        
        
        
#%%
        
li = np.zeros((1, 1000))[0]
li[np.random.randint(0, 1000, 500)] = 1
bil = NBEnum('test', li)
bil._encoding()
