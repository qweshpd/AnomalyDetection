#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm

text = list(pd.read_csv(r'.Messages2.csv')['Message'])

dicValues = {}
dicVectorValues = {}
tmpAllArr = []
for devItem in text:
    tmpArr = devItem.split(':')[3:]
    dicValues[':'.join(devItem.split(':')[:3])] = tmpArr
    tmpAllArr.append(tmpArr)

tmpAllArr = np.array(tmpAllArr)
_, feat = tmpAllArr.shape
kindArr = [list(set(tmpAllArr[:, i])) for i in range(feat)]

for indexKind in range(len(kindArr)):
    kindArr[indexKind].sort()

tmpAllArr = []
tmpMapKeyArr = []

for key in dicValues:
    dicVectorValues[key] = []
    kindIndex = 0
    for arrItem in dicValues[key]:
        oneFeatureArr = np.zeros(len(kindArr[kindIndex]))
        oneFeatureArr[kindArr[kindIndex].index(arrItem)] = 1
        dicVectorValues[key] += oneFeatureArr.tolist()
        kindIndex += 1
    tmpAllArr.append(dicVectorValues[key])
    tmpMapKeyArr.append(key)

tmpAllArr = np.array(tmpAllArr)
tmpMapKeyArr = np.array(tmpMapKeyArr)
errorRate = 0.05

while abs(errorRate):
    clf = svm.OneClassSVM(nu = errorRate, 
                          kernel = "rbf", gamma = 0.1, tol = 1e-9)
    clf.fit(tmpAllArr)
    y_pred_train = clf.predict(tmpAllArr)
    
    errorArr = np.where(y_pred_train == -1)[0]
    if len(errorArr) < 0.1 *len(y_pred_train):
        break
    
    errorRate -= 0.01

pca = PCA(n_components = feat)
pca.fit(tmpAllArr)

print( "PCA:" + str(pca.explained_variance_ratio_) )
print( "rate: " + str(errorRate) )    
print(str(len(y_pred_train)) + ":" + str(len(errorArr)))

for item in tmpMapKeyArr[errorArr]:
    print( "Error:" + str(item))
