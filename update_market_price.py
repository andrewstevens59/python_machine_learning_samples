import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from sklearn.cluster import KMeans
from numpy import linalg as LA

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import json
import copy

import math
import sys
import re

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import string
import random as rand

import os
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from uuid import getnode as get_mac
import socket
import json
import mysql.connector
import psutil


def checkIfProcessRunning(processName, command):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 3 and processName.lower() in cmdline[2] and command in cmdline[3]:
                count += 1
            elif len(cmdline) > 2 and processName.lower() in cmdline[1] and command in cmdline[2]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 2:
        sys.exit(0)


checkIfProcessRunning("update_market_price.py", "")


cnx = mysql.connector.connect(user='random', password='random',
                              host='random',
                              database='demand_market')

cursor = cnx.cursor()
query = ("SELECT sum(demand_allocation - 50) as net_demand, market_id FROM user_demand group by market_id")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
    setup_rows.append(row1)


demands = []
market_z_score = {}
for row in setup_rows:

    print (row)
    demands.append(row[0])

mean = float(np.mean(demands))
std = float(np.std(demands))

for row in setup_rows:

    market_z_score[row[1]] = (float(row[0]) - mean) / std

print (market_z_score)

cursor = cnx.cursor()
   
for row in setup_rows:

    net_demand = row[1]
    market_id = row[1]

    query = ("SELECT * FROM current_market_price where market_id='{0}' limit 1".format(market_id))

    cursor.execute(query)

    sample_rows = []
    for row1 in cursor:
        sample_rows.append(row1)
        break

    if len(sample_rows) > 0:
        market_price = sample_rows[0][1]
        query = ("UPDATE current_market_price set market_price='{0}', timestamp=now() where market_id='{1}'".format(market_z_score[market_id], market_id))
        cursor.execute(query)
        cnx.commit()

    else:
        query = ("INSERT into current_market_price (market_id, market_price, timestamp) value ('{0}', '{1}', now())".format(market_id, market_z_score[market_id]))
        cursor.execute(query)
        cnx.commit()

        query = ("INSERT into market_price_history (market_id, market_price, timestamp) value ('{0}', '{1}', now())".format(market_id, market_z_score[market_id]))
        cursor.execute(query)
        cnx.commit() 


query = ("SELECT  t1.market_id, t2.market_price FROM market_price_history t1, current_market_price t2, (select max(timestamp) as max_timestamp, market_id from market_price_history t4 group by t4.market_id) as t3 where t1.market_id=t2.market_id and t1.timestamp = t3.max_timestamp and t3.market_id=t2.market_id and abs(t2.market_price-t1.market_price) > 0.1")
cursor = cnx.cursor()
cursor.execute(query)

sample_rows = []
for row1 in cursor:
    sample_rows.append(row1)

for row1 in sample_rows:
    print "update"
    query = ("INSERT into market_price_history (market_id, market_price, timestamp) value ('{0}', '{1}', now())".format(row1[0], row1[1]))
    cursor.execute(query)
    cnx.commit()  


query = ("UPDATE user_demand t1, current_market_price t2 set t1.open_price=t2.market_price where t1.open_price is null and t1.market_id=t2.market_id")
cursor.execute(query)
cnx.commit()


query = ("UPDATE user_demand t1, current_market_price t2 set t1.pnl=(t2.market_price - t1.open_price) * (t1.demand_allocation - 50) where t1.market_id=t2.market_id")
cursor.execute(query)
cnx.commit()



cursor.close()