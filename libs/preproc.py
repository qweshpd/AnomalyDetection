#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Collection of necessary preprocessing method.
'''

import datetime
import pandas as pd
import numpy as np

def normalize(data, method):
    if method == '01':
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == '-11':
        return (data - np.mean(data)) / (np.max(data) - np.min(data))