# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm
import copy
#from itertools import groupby

_SCALE = 15
_THRESHOLD = 0.5
_RARE_FACTOR = 0.005

class auto_onehot(object):
    """Encode categorical integer features using one-hot.
    
    Parameters
    --------
    featuredic : dict
        All features with feature name.    
    
    Examples
    --------
    
    >>>feature = {"fe1": ["ab", "bc", "ca"],
                  "fe2": [10*i for i in np.arange(8)]}
    >>>auto_onehot(feature).transform(["ab", 60, "Sun"])
    array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.])
    """
    
    def __init__(self, featuredic):
        self.feature = list(featuredic.keys()) # All features
        self.fnum = len(featuredic.keys()) # total feature number
        self.attrnum = [] # number of each feature
        for fea in featuredic.keys():
            setattr(self, fea, featuredic[fea])
            self.attrnum.append(len(featuredic[fea]))
        self.vnum = sum(self.attrnum) # vector length
    
    def transform(self, slicefeature):
        """Fit OneHotEncoder to X, then transform X.
        
        Parameters
        ----------
        X : array-like, list of feature instances.
            Input array of features.
        """
        assert len(slicefeature) == self.fnum, "please input only %d feature(s)"%self.fnum
        feature_array = np.zeros((1, self.vnum))[0]
        for i in np.arange(self.fnum):
            allvalue = getattr(self, self.feature[i])
            tmpnum = sum(self.attrnum[:i]) + allvalue.index(slicefeature[i])
            feature_array[tmpnum] = 1
        return feature_array


class AnalyzeEnum(object):
    """Automatical anomaly detection for enum-like variables."""
    
    def __init__(self, model=None):
        self.model = model

    def _encoding(self, all_data):
        """Encode categorical integer features using one-hot."""
        tranf = self.encode.transform
        code_array = np.vstack(list(map(lambda x:tranf([x]), all_data)))
        
        return code_array
    
    def _seq_analy(self, code, value=1):
        """Analysis sequential code.
        
        Parameters
        ----------
        code : array-like, contains only 0 or 1
            Code data to be analyzed.
        value : int, 0 or 1
            Target value.
        """
        isvalue = np.concatenate(([0], np.equal(code, value), [0]))
        inds = np.where(np.abs(np.diff(isvalue)) == 1)[0].reshape(-1, 2)
        last = inds[:, 1] - inds[:, 0]
        inds[:, 1] = inds[:, 1] - 1
        return np.vstack((inds[:, 0], inds[:, 1], last)).T
    
    def _freq_anal(self, nlist):
        """Statistical matrix."""
        dmean = nlist.mean()
        dstd = nlist.std()
        factor = np.sqrt(-np.log(1 - _THRESHOLD)*_SCALE)
        dmax = dmean + dstd * factor
        dmin = max(0, dmean - dstd * factor)

        return  dmean, dstd, dmax, dmin
    
    def _modeling(self, var_name, datalist):
        """Build statistical model based on historical data."""
        self.name = var_name
        self.data = datalist
        features = list(set(datalist))
        features.sort()
        self.encode = auto_onehot({var_name: features})
        code_array = self._encoding(datalist)    
        
        tmpmodel = pd.DataFrame(columns=features, 
                          index = ["mu", "sigma", "tmax", "tmin"])
        self.feature_array = {}
        for i in np.arange(self.encode.attrnum[0]):
            tmparray = self._seq_analy(code_array[:, i])
            self.feature_array[features[i]] = tmparray
            tmpmodel[features[i]] = self._freq_anal(tmparray[:, 2])
        
        if self.model is None:
            self.model = tmpmodel
            
        self.features = features
        self.code_array = pd.DataFrame(code_array, columns=features)
        return code_array
    
    def getscore(self, data=None):
        """Analyze data based on model.
        
        Parameters
        ----------
        data : 
            Data to be analyzed.
        
        Returns
        -------
        inds : 2D array-like [indi, indf]
            Initial and final indices of data detected as anomaly.
        """
        features = self.features
        if data is None:
            encode_data = self._encoding(self.data)
            tmpdict = self.feature_array
        else:
            tmpdict = {}
            encode_data = self._encoding(data)
            for i in np.arange(self.encode.attrnum[0]):
                tmparray = self._seq_analy(encode_data[:, i])
                tmpdict[features[i]] = tmparray
            
        model = self.model
        final_score = np.zeros((1, encode_data.shape[0]))
        scoredict = {}
        for onef in features:
            model_array = np.array(model[onef])
            mean = model_array[0]
            std = model_array[1] if model_array[1] else 1
            test_array = tmpdict[onef][:, 2]

            score_array = 1 - np.exp(-((test_array-mean)/std)**2/_SCALE)
            scoredict[onef] = np.vstack((tmpdict[onef][:, 0], 
                                         tmpdict[onef][:, 1],
                                         tmpdict[onef][:, 2],
                                         score_array)).T
        
            for i in np.arange(len(test_array)):
                indif = tmpdict[onef][:, :2][i]
                score = score_array[i]
                final_score[0, indif[0]: indif[1] + 1] = score
        
        finalscore = pd.DataFrame(np.hstack((encode_data, final_score.T)),
                                  columns=self.features+["score"])
                                  
        return scoredict, finalscore
    
    def histanalyze(self, data=None, show=False):
        """Analyze data based on model.
        
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
        """
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
        alert = {}
        for onef in features:
            model_array = np.array(model[onef])
            test_array = tmpdict[onef][:, 2]
            tmp = np.logical_or(test_array > model_array[2],
                                test_array < model_array[3])
            inds = np.where(tmp)[0]
            
            if len(inds):
                alert[str(onef)] = tmpdict[onef][inds][:, :2]
            
        if show:
            try:
                import matplotlib.pylab as plt
                plt.rcParams["figure.figsize"] = 15, 6
            except ImportError:
                print("matplotlib is not available.")
            else:
                plt.figure()
                for fi in features:
                    entry = features.index(fi)
                    pltmodel = np.array(model[fi])
                    test_array = tmpdict[fi][:, 2]
                    base = np.arange(max(-10, min(test_array)-20), 
                                     max(test_array)+20, 1)
                    normal = norm.pdf(base, pltmodel[0], pltmodel[1] \
                                      if pltmodel[1] else 1)
                    anomalous = np.logical_or(base > pltmodel[2], 
                                              base < pltmodel[3])
                    
                    plt.subplot(len(features), 1, entry + 1)
                    plt.hist(test_array, bins=base-0.5, 
                             normed=True, zorder=1)
                    plt.fill_between(base, normal, where=anomalous, 
                                     color=[1,0,0,0.4], zorder=2)
                    plt.plot(base, normal, color="black", zorder=3)
                    plt.title("Feature distribution of %s"%fi, fontsize = 20)
                    plt.show()
        
        return alert


class NBEnum(AnalyzeEnum):
    
    def _seq_analy(self, code, value=1):
        """Analysis sequential code.
        
        Parameters
        ----------
        code : array-like, contains only 0 or 1
            Code data to be analyzed.
        value : int, 0 or 1
            Target value.
        """
        isvalue = np.concatenate(([0], np.equal(code, value), [0]))
        inds = np.where(np.abs(np.diff(isvalue)) == 1)[0].reshape(-1, 2)
        inds[:, 1] = inds[:, 1] - 1
        last = [i.seconds for i in (self.timelist[inds[:, 1]] - self.timelist[inds[:, 0]])]
        return np.vstack((inds[:, 0], inds[:, 1], last)).T
    
    def preprocess(self, CData):
        """Extract data info from DataItem and prepare for processing"""
        var = CData.var_name
        data = np.vstack(CData.data)
        data_array = data[:, 1]
        time_array = data[:, 0]
        self.timelist = time_array
        _ = self._modeling(var, data_array)
        
        return time_array, data_array
    
    def process(self, data=None):
        """Main process of algorithm which analyze each feature."""
        _, score_array = self.getscore(data = data)
        return score_array
    
    def postprocess(self, score, time, csc=False):
        """Recover processed data into uniformed format."""
        datalist = self.data
        score_array = np.array(score["score"])
        if csc:
            self._csc(score_array)
        return pd.DataFrame(np.vstack((time, datalist, score_array)).T,
                            columns=["timestamp", "data", "score"])
    
    def analyze(self, DataItem, csc=False):
        """Automatically processing data."""
        time_array, data_array = self.preprocess(DataItem)
        score_array = self.process(data = data_array)
        result = self.postprocess(score_array, time_array, csc=csc)
        return result
    
    def _csc(self, score_array):
        """Consider speacial cases for analyzing. Including:
        a) Rare change 
        b) Rare feature
        c) Change of frequncy
        
        Parameters
        ----------
        score_array : array-like
            Score of data.
        """
        code_array = np.array(self.code_array).T
        initial_score = copy.deepcopy(score_array[0])
        final_score = copy.deepcopy(score_array[-1])
        f_array = self.feature_array
        tmp_factor = min(10, np.ceil(len(self.data) * _RARE_FACTOR))
        
        # Rare change 
        for key in f_array:
            c_array = f_array[key][:, :2]
            if c_array.shape[0] < tmp_factor:
                for indi, indf in c_array:
                    score_array[indi] = 1
                    score_array[indf] = 1
        
        # Rare feature            
        for tmp in code_array:
            if tmp.sum() < tmp_factor:
                score_array[np.where(tmp)[0]] = 1
        
        # Change of frequncy
        if len(self.features) > 2:
            d = np.array(self.code_array)
            y = []
            for i in np.arange(len(d)-1):
                y.append(sum(abs(d[i]- d[i + 1])))
                
            tp = AnalyzeEnum()
            _ =  tp._modeling("test", y)
            ana = tp.histanalyze()["0.0"].reshape(1, -1)[0]
            for j in ana:
                score_array[j] = 1
        
        score_array[0] = initial_score
        score_array[-1] = final_score
        # TODO: Merge initial and final indices.
        return score_array
    
    def get_cache(self, score_array):
        """Get data to cache."""
        return self.model
