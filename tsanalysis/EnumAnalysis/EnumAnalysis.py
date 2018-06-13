# -*- coding: utf-8 -*-

import logging
import copy
import numpy as np
import pandas as pd
from scipy.stats import norm

from .encode import auto_onehot

logger = logging.getLogger("EnumAnalysis")

_SCALE = 15
_THRESHOLD = 0.8
_RARE_FACTOR = 0.005
_cache_index = ["mu", "sigma", "tmax", "tmin", "scale", "rfactor"]

def dd_convert(inst, target='dict'):
    """Convert model from DataFrame to dict, or the other way."""
    t = type(inst)
    if target == 'dict':
        if t == dict:
            return inst
        elif t == pd.core.frame.DataFrame:
            ddict = {}
            all_features = list(inst.columns)
            indices = list(inst.index)
            cache_num = len(_cache_index)
            for onef in all_features:
                tmp = np.array(inst[onef])
                ddict[onef] = dict([(indices[i], tmp[i]) for i in np.arange(cache_num)])
            return ddict
        else:
            raise ValueError("Please input a valid data format!")
    elif target == 'df':
        if t == dict:
            all_features = list(inst.keys())
            dframe = pd.DataFrame(columns=all_features, index=_cache_index)
            for onef in all_features:
                for onevalue in inst[onef]: 
                    dframe.loc[[onef],[onevalue]] = inst[onef][onevalue]
            return dframe
        elif t == pd.core.frame.DataFrame:
            return inst
        else:
            raise ValueError("Please input a valid data format!")
    else:
        raise ValueError("Please input a valid target type!")


class AnalyzeEnum(object):
    """Automatical anomaly detection for enum-like variables."""
    
    def __init__(self, model=None):
        if model is None:
            self.model = None
        else:
            self.model = dd_convert(model, target='df')
            

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

        return  dmean, dstd, dmax, dmin, _SCALE, _RARE_FACTOR
    
    def _modeling(self, var_name, datalist):
        """Build statistical model based on historical data."""
        self.name = var_name
        self.data = datalist
        features = list(set(datalist))
        features.sort()
        self.encode = auto_onehot({var_name: features})
        code_array = self._encoding(datalist)    
        
        tmpmodel = pd.DataFrame(columns=features, index = _cache_index)
        self.feature_array = {}

        
        if self.model is None:
            for i in np.arange(self.encode.attrnum[0]):
                tmparray = self._seq_analy(code_array[:, i])
                self.feature_array[features[i]] = tmparray
                tmpmodel[features[i]] = self._freq_anal(tmparray[:, 2])
            self.model = tmpmodel
        else:
            for i in np.arange(self.encode.attrnum[0]):
                tmparray = self._seq_analy(code_array[:, i])
                self.feature_array[features[i]] = tmparray
            
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
#            score_array = np.exp(-0.5 * ((test_array-mean)/std)**2)/(np.sqrt(2*np.pi) * std)
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

class CoFEnum(AnalyzeEnum):
    
    def __init__(self, timelist):
        self.timelist = timelist
        self.model = None
    
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
        last = [i.days*3600*24 + i.seconds for \
                i in (self.timelist[inds[:, 1]] - self.timelist[inds[:, 0]])]
        return np.vstack((inds[:, 0], inds[:, 1], last)).T
    
    
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
        last = [i.days*3600*24 + i.seconds for \
                i in (self.timelist[inds[:, 1]] - self.timelist[inds[:, 0]])]
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
    
    def _modify(self, score, time, csc=False):
        """Recover processed data into uniformed format."""
        datalist = self.data
        score_array = np.array(score["score"])
        reason = ["distribution score"]*len(score_array)
        if csc:
            score_array, reason = self._csc(score_array, reason)
        return pd.DataFrame(np.vstack((time, datalist, score_array, reason)).T,
                            columns=["timestamp", "data", "score", "reason"])
    
    def analyze(self, DataItem, csc=False):
        """Automatically processing data."""
        time_array, data_array = self.preprocess(DataItem)
        score_array = self.process(data = data_array)
        result = self._modify(score_array, time_array, csc=csc)
        return result
    
    def _csc(self, score_array, reason):
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
                    reason[indi] = "Initial Rare change"
                    reason[indf] = "Final Rare change"
        
        # Rare feature            
        for tmp in code_array:
            if tmp.sum() < tmp_factor:
                tmpind = np.where(tmp)[0]
                score_array[tmpind] = 1
                for i in tmpind:
                    reason[i] = "Rare feature"
        
        # Change of frequncy
        if len(self.features) > 2:
            d = np.array(self.code_array)
            y = []
            for i in np.arange(len(d)-1):
                y.append(sum(abs(d[i]- d[i + 1])))
            tp = CoFEnum(self.timelist[:-1])
            _ =  tp._modeling("test", y)
            try:
                ana = tp.histanalyze()["0.0"].reshape(1, -1)[0]
            except:
                pass
            else:
                for j in ana:
                    score_array[j] = 1
                    reason[j] = "Change of frequncy"
        score_array[0] = initial_score
        score_array[-1] = final_score
        reason[0] = "distribution score"
        reason[-1] = "distribution score"
        return score_array, reason
    
    def get_cache(self):
        """Get data to cache."""
        return dd_convert(self.model, target='dict')

