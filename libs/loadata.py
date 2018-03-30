#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Load data from MongoDB.'''

import datetime
import pandas as pd
import pymongo
from pymongo import MongoClient


def getdata(devname, intf, sdate = None, edate = None):
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
        
    start_date = _getdtime(sdate)
    end_date = _getdtime(edate)


    interf = _recintf(intf)
    # extract data from database
    

    mongoClient = MongoClient('mongodb://%s:%s@localhost:27017' % 
                          ("username", "password"))
    MLDataQappDB = mongoClient["MLDataQappNetbrain"]
    dbCollection = MLDataQappDB["InterfaceDetail"]
    cursor = dbCollection.find({"device_name": devname, 
                "date": {
                    "$gte": start_date,
                    "$lt": end_date
                }}).sort([("date", pymongo.ASCENDING)])
 
    timeArr = []
    valueArr = []
    for itemCursor in cursor:
        timeArr.append(itemCursor["date"] )
        # devName = itemCursor["device_name"]
        InterfaceDetailArr = itemCursor["InterfaceDetail"]
        for oneInf in InterfaceDetailArr:
            infName = oneInf["intf"]
            if not infName == interf:
                continue
            
            input_rate = float(oneInf["input_rate_bit"])
            valueArr.append(input_rate)   
        
    timeseries = pd.Series(valueArr, index = timeArr)
    
    return timeseries

def _getdtime(ymd):
    '''
    Convert YYYY-MM-DD to datetime.datetime format.
    '''
    y, m, d  = map(int, ymd.split('-'))
    return datetime.datetime(y, m, d, 0, 0, 0)

def _recintf(shortintf):
    '''
    Convert short interface name to its full name.
    '''
    temp = shortintf.split('/')
    if temp[0].startswith('f'):
        temp[0] = 'FastEthernet' + temp[0][-1] 
    elif temp[0].startswith('te'):
        temp[0] = 'TenGigabitEthernet' + temp[0][-1]
    elif temp[0].startswith('g'):
        temp[0] = 'GigabitEthernet' + temp[0][-1]
    return '/'.join(temp)

if __name__ == "__main__":
    pass
