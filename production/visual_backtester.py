import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from pytz import timezone
import xgboost as xgb
from datetime import timedelta
import mysql.connector
from lxml.html import fromstring
from itertools import cycle
from scipy import stats
import datetime as dt
import traceback


import re

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
import pycurl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json


import os
import bisect

import paramiko
import json

import logging
import os
import enum

import matplotlib

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

def is_valid_trading_period(ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    if s.weekday() in {4} and s.hour in {21, 22, 23}:
        return False
    if s.weekday() in {5}:
        return False
    if s.weekday() in {6} and s.hour < 21:
        return False
    
    return True

def get_time_series(symbol, time, granularity="H1"):

    response_buffer = StringIO()
    curl = pycurl.Curl()

    curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=" + granularity + "&alignmentTimezone=America%2FNew_York")

    curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

    curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

    curl.perform()
    curl.close()

    response_value = response_buffer.getvalue()
    j = json.loads(response_value)['candles']

    prices = []
    times = []


    for index in range(len(j)):
        item = j[index]

        s = item['time']
        s = s[0 : s.index('.')]
        timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

        if is_valid_trading_period(timestamp):
            times.append(timestamp)
            prices.append(item['closeMid'])
  

    return prices, times

def checkIfProcessRunning(processName, command):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 1 and processName.lower() in cmdline[1]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 2:
        sys.exit(0)

if get_mac() != 150538578859218:
    root_dir = "/root/" 
else:
    root_dir = "" 


class MyFormatter(logging.Formatter):
    converter=dt.datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

formatter = MyFormatter(fmt='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def load_time_series(symbol, year, is_bid_file):

    if get_mac() == 150538578859218:
        prefix = '/Users/andrewstevens/Downloads/economic_calendar/'
    else:
        prefix = '/root/trading_data/'

    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(prefix) if isfile(join(prefix, f))]

    pair = symbol[0:3] + symbol[4:7]

    for file in onlyfiles:

        if pair in file and 'Candlestick_1_Hour_BID' in file:
            break

    if pair not in file:
        return None

    with open(prefix + file) as f:
        content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 

    from_zone = tz.gettz('America/New_York')
    to_zone = tz.tzutc()

    prices = []
    highs = []
    lows = []
    times = []
    volumes = []

    content = content[1:]

    if year != None:
        start_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".1.1 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())
        end_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".12.31 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())

    for index in range(len(content)):

        toks = content[index].split(',')
        utc = datetime.datetime.strptime(toks[0], "%d.%m.%Y %H:%M:%S.%f")

        time = calendar.timegm(utc.timetuple())

        if year == None or (time >= start_time and time < end_time):

            high = float(toks[2])
            low = float(toks[3])
            o_price = float(toks[1])
            c_price = float(toks[4])
            volume = float(toks[5])

            if high != low or volume > 0:
                prices.append(o_price)
                times.append(time)
                volumes.append(volume)
                highs.append(high)
                lows.append(low)


    return prices, times, volumes, lows, highs

def create_correlation_graph():

    count = 0
    for i, compare_pair in enumerate(currency_pairs):

        prices, times, volumes, lows, highs = load_time_series(compare_pair, None, False)
        before_price_df2 = pd.DataFrame()
        before_price_df2["prices" + str(i)] = prices
        before_price_df2["times"] = times
        count += 1

        if count > 1:
            before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
            before_all_price_df.reset_index(inplace=True)
        else:
            before_all_price_df = before_price_df2

    

    times = before_all_price_df["times"].values.tolist()
    hours_back = 20 * 24
    equity = 0

    for iteration in range(1000):

        for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:

            print (20 * 24, len(before_all_price_df) - (20 * 24))
            start_time = rand.randint(20 * 24, len(before_all_price_df) - (20 * 24))
            
            sorted_pairs = []
            for i, compare_pair in enumerate(currency_pairs):
                if select_currency not in compare_pair:
                    continue

                after_prices = before_all_price_df["prices" + str(i)].values.tolist()[start_time:start_time+(20*24)]

                pair1 = before_all_price_df["prices" + str(i)].values.tolist()[start_time-hours_back:start_time]
                display_pair = compare_pair
                if compare_pair[0:3] != select_currency:
                    pair1 = [1.0 / price for price in pair1]
                    after_prices = [1.0 / price for price in after_prices]
                    display_pair = compare_pair[4:7] + "_" + compare_pair[0:3]

                mean = np.mean(pair1)
                std = np.std(pair1)

                pair1 = [(p - mean) / std for p in pair1]

                pip_size = 0.0001
                if "JPY" in compare_pair:
                    pip_size = 0.01
                    continue


                sorted_pairs.append([pair1[-1], after_prices, std, pip_size])


                #plt.plot(range(len(pair1)), pair1, label="Pair {} {}".format(display_pair, (after_prices[-1] - after_prices[0])))

            try:
                sorted_pairs = sorted(sorted_pairs, key=lambda x: x[0])

                total_profit = 0
         
                for item in [sorted_pairs[0], sorted_pairs[1]]:

                    profit = (item[1][-1] - item[1][0]) / item[2]
                    
                    for price in item[1]:

                        if abs(price - item[1][0]) / item[2] > 32:
                            profit = (price - item[1][0]) / item[2]
                            break
                    

                    total_profit += profit / abs(sorted_pairs[-1][0] - sorted_pairs[0][0])
                    total_profit -= ((item[3] * 5) / item[2]) / abs(sorted_pairs[-1][0] - sorted_pairs[0][0])

                for item in [sorted_pairs[-1], sorted_pairs[-2]]:

                    profit = (item[1][-1] - item[1][0]) / item[2]
                    
                    for price in item[1]:

                        if abs(price - item[1][0]) / item[2] > 32:
                            profit = (price - item[1][0]) / item[2]
                            break
                    

                    total_profit -= profit / abs(sorted_pairs[-1][0] - sorted_pairs[0][0])
                    total_profit -= ((item[3] * 5) / item[2]) / abs(sorted_pairs[-1][0] - sorted_pairs[0][0])
            except:
                pass 
        
            '''
            plt.title("Related Pairs {} Profit {}".format(sselect_currency, profit))
            plt.legend()
            plt.show()
            '''

            equity += total_profit
            print ("total profit", equity)


create_correlation_graph()




