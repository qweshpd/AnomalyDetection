#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import re
import pymongo
from pymongo import MongoClient

#%% crawler

url = "https://www.cisco.com/en/US/docs/internetworking/troubleshooting/guide/tr1901.html"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

wbdata = requests.get(url, headers = headers).text  

soup = BeautifulSoup(wbdata,'html.parser')

content = soup.find_all(name = 'p', attrs = {'class':re.compile('^pB')})
#content = soup.find_all(['p', re.compile('^h')])
temp = []
for te in content:
    temp.append(te.get_text())
    print(te.get_text())


#%% MongoDB
     
mongoClient = MongoClient('mongodb://%s:%s@%s:27017' % 
                      (username, password, hostname))
TBDB = mongoClient["TroubleShooting"]
dbCollection = TBDB["Cisco"]

tbinfo = {
            'content': temp,
            'url': url
         }

result = dbCollection.insert_one(tbinfo)


