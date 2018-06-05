import numpy as np

class auto_onehot(object):
    '''
    Encode categorical integer features using one-hot.
    
    
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
        assert len(slicefeature) == self.fnum
        feature_array = np.zeros((1, self.vnum))[0]
        for i in np.arange(self.fnum):
            allvalue = getattr(self, self.feature[i])
            tmpnum = sum(self.attrnum[:i]) + allvalue.index(slicefeature[i])
            feature_array[tmpnum] = 1
        return feature_array