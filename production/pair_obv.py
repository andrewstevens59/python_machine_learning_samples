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

    open_prices = []
    close_prices = []
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
                open_prices.append(o_price)
                close_prices.append(c_price)
                times.append(time)
                volumes.append(volume)
                highs.append(high)
                lows.append(low)


    return open_prices, close_prices, times, volumes, lows, highs

class Order():

    def __init__(self):
        self.amount = 0



item_ranking = {}
for pair in ["GBP_CAD"]:#currency_pairs[2:]:
    open_prices1, close_prices1, times1, volumes1, lows1, highs1 = load_time_series(pair, None, False)

    orders = []
    equity = 0
    equity_curve = []
    for index in range(24 * 20, len(open_prices1), 24):

        open_prices = open_prices1[index - (24 * 20):index]
        close_prices = close_prices1[index - (24 * 20):index]
        times = times1[index - (24 * 20):index]
        volumes = volumes1[index - (24 * 20):index]
        lows = lows1[index - (24 * 20):index]
        highs = highs1[index - (24 * 20):index]

        item_ranking[pair] = []
        print (len(open_prices), "**")

        for time_frame in range(len(open_prices)):
            obv = 0
            price = 0
            price_deltas = []
            obv_deltas = []
            for o, c, v, t in zip(open_prices[time_frame:], close_prices[time_frame:], volumes[time_frame:], times[time_frame:]):

                if c > o:
                    obv += v
                    price += abs(c - o)
                else:
                    obv -= v
                    price -= abs(c - o)

                obv_deltas.append(obv)
                price_deltas.append(price)

            z_score_obv = (obv_deltas[-1] - np.mean(obv_deltas)) / np.std(obv_deltas)
            z_score_price = (price_deltas[-1] - np.mean(price_deltas)) / np.std(price_deltas)

            if np.std(obv_deltas) > 0 and np.std(price_deltas) > 0:
                item_ranking[pair].append(z_score_obv - z_score_price)

        if len(item_ranking[pair]) == 0:
            continue

        entry = np.percentile(item_ranking[pair], 50)

        if abs(entry) > 1:
            order = Order()
            order.open_price = close_prices[-1]
            order.amount = 1
            order.dir = entry > 0
            order.pair = pair
            orders.append(order)

            '''
            order = Order()
            order.open_price = close_prices[-1]
            order.amount = 1
            order.dir = False
            order.pair = pair
            orders.append(order)
            '''

        new_orders = []
        float_profit = 0
        total_pips = 0
        profit = 0

        existing_order_count = 0
        for order in orders:

            if (close_prices[-1] > order.open_price) == order.dir:
                profit = abs(close_prices[-1] - order.open_price)
            else:
                profit = -abs(close_prices[-1] - order.open_price)

            pip_size = 0.0001
            if "JPY" in order.pair:
                pip_size = 0.01

            profit /= pip_size
            profit -= 5

            
            if (close_prices[-1] > order.open_price) == order.dir:
                profit_low = abs(close_prices[-1] - order.open_price)
            else:
                profit_low = -abs(close_prices[-1] - order.open_price)

            if (close_prices[-1] > order.open_price) == order.dir:
                profit_high = abs(close_prices[-1] - order.open_price)
            else:
                profit_high = -abs(close_prices[-1] - order.open_price)

            profit_low /= pip_size
            profit_high /= pip_size
            
            
            if min(profit_high, profit_low) < -50:
                print ("stop")

                #total_orders = len([order.pair for order in orders if order.pair == item[0]])

                pip_size = 0.0001
                if "JPY" in pair:
                    pip_size = 0.01

                new_order = Order()
                new_order.pair = order.pair
                new_order.dir = (order.dir == False)
                new_order.is_revert = False
                new_order.open_price = close_prices[-1]
                new_order.amount = order.amount * 1.5
                new_order.open_time = index
                new_orders.append(new_order)

                '''
                new_order = Order()
                new_order.pair = order.pair
                new_order.dir = (order.dir == True)
                new_order.is_revert = False
                new_order.open_price = close_prices[-1]
                new_order.amount = order.amount 
                new_order.open_time = index
                new_orders.append(new_order)
                '''

                total_pips -= 6
                

                equity += -50 * order.amount 
                continue
            
            '''
            if (entry > 0) != order.dir:
                equity += profit * order.amount 
                continue
            '''

            total_pips += profit 

            profit *= order.amount 
            
            float_profit += profit
            new_orders.append(order)

        orders = new_orders
        print (pair, "orders", len(orders))
        print (pair, "equity", equity + float_profit)
        print (pair, "total pips", total_pips)
        equity_curve.append(equity + float_profit)

        all_returns = [b - a for a, b in zip(equity_curve[:-1], equity_curve[1:]) if abs(a - b) > 0]
        sharpe = np.mean(all_returns) / np.std(all_returns)
        print (pair, "sharpe", sharpe)
        
        
        if total_pips >= 400 and float_profit > 0:
            orders = new_orders
            equity += float_profit 
            orders = []
            float_profit = 0
        





