#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Load data from MongoDB.'''

import datetime
import pandas as pd
import pymongo
from pymongo import MongoClient

def getdata(devname, intf, sdate = None, edate = None, dbname, collection, feature):
    '''
    Load data from MongoDB.
    
    Parameters
    ----------
    devname: string
        device name
    intf : string
        interface name
    startdate : string, date format YYYY-MM-DD, optional (default = Nov 18, 2017)
        start date of data loaded
    endate : string, date format YYYY-MM-DD, optional (default = last calender day)
        end date of data loaded
    
    Returns
    -------
    timeseries : pd.Series data
    '''
    
    if not sdate:
        sdate = '2017-11-18'

    if not edate:
        edate = datetime.date.today().strftime('%Y-%m-%d')
        
    start_date = datetime.datetime.strptime(sdate, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(edate, '%Y-%m-%d')


    interf = _recintf(intf)
    # extract data from database
    
    cursor = MongoClient('mongodb://%s:%s@%s:27017' % 
                        ('username', 'password', 'host'))\
                        [dbname][collection]\
                        .find({'device_name': devname, 
                        'date': {
                            '$gte': start_date,
                            '$lt': end_date
                        }}).sort([('date', pymongo.ASCENDING)])
    timeArr = []
    valueArr = []
    for itemCursor in cursor:
        timeArr.append(itemCursor['date'] )
        # devName = itemCursor['device_name']
        InterfaceDetailArr = itemCursor[collection]
        for oneInf in InterfaceDetailArr:
            infName = oneInf['intf']
            if not infName == interf:
                continue
            
            input_rate = float(oneInf[feature])
            valueArr.append(input_rate)           
    timeseries = pd.DataFrame({'data': valueArr, 'time' : timeArr},
                                 columns = ['time', 'data'])
    
    return timeseries

def _recintf(shortintf):
    '''
    A naive replacement of Interface API. Convert short interface name to 
    its full name. 

    '''
    temp = shortintf.split('/')
    if temp[0].startswith('f'):
        temp[0] = 'FastEthernet' + temp[0][-1] 
    elif temp[0].startswith('te'):
        temp[0] = 'TenGigabitEthernet' + temp[0][-1]
    elif temp[0].startswith('g'):
        temp[0] = 'GigabitEthernet' + temp[0][-1]
    return '/'.join(temp)
