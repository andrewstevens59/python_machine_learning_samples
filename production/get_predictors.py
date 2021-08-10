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

class Order():

    def __init__(self):
        self.amount = 0

def create_correlation_graph():

    count = 0
    for i, compare_pair in enumerate(currency_pairs):

        print (compare_pair)
        prices, times = get_time_series(compare_pair, 24 * 35)
        before_price_df2 = pd.DataFrame()
        before_price_df2["prices" + str(i)] = prices
        before_price_df2["times"] = times
        count += 1

        if count > 1:
            before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
            before_all_price_df.reset_index(inplace=True)
        else:
            before_all_price_df = before_price_df2

    before_all_price_df.sort_values(by='times', ascending=True, inplace=True)

    times = before_all_price_df["times"].values.tolist()

    delta_window = []
    days_back_range = 20
    back_days = 20
    delta_final_map = {}

    sorted_pairs = []
    for j, currency_pair in enumerate(currency_pairs):

        delta_map = {}
        correlation_map = {}
        for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:

            if select_currency not in currency_pair:
                continue

            delta_map[select_currency] = {}
            correlation_map[select_currency] = {}

            for days_back in range(2, days_back_range):
                hours_back = days_back * 24

                z_scores = []
                z_score_map = {}

                pair2 = before_all_price_df["prices" + str(j)].values.tolist()[-hours_back:]

                if currency_pair[0:3] != select_currency:
                    pair2 = [1.0 / price for price in pair2]

                correlations = []
                for i, compare_pair in enumerate(currency_pairs):
                    if select_currency not in compare_pair:
                        continue

                    pair1 = before_all_price_df["prices" + str(i)].values.tolist()[-hours_back:]
                    if compare_pair[0:3] != select_currency:
                        pair1 = [1.0 / price for price in pair1]

                    if i != j:
                        correlation, p_value = stats.pearsonr(pair1, pair2)
                        correlations.append(correlation)

                    mean = np.mean(pair1)
                    std = np.std(pair1)

                    z_score = (pair1[-1] - mean) / std 
                    z_scores.append(z_score)

                    z_score_map[compare_pair] = z_score

                mean_correlation = np.mean(correlations)
                mean_z_score = np.mean(z_scores)
                delta = z_score_map[currency_pair] - mean_z_score

                if currency_pair[0:3] != select_currency:
                    delta = -delta

                delta_map[select_currency][days_back] = delta
                correlation_map[select_currency][days_back] = mean_correlation


        matrix = np.zeros((days_back_range, days_back_range))

        diffs = []
        diff_left = []
        diff_right = []
        for i in range(2, days_back_range):
            for j in range(2, days_back_range):

                delta_i = delta_map[currency_pair[0:3]][i]
                delta_j = delta_map[currency_pair[4:7]][j]

                correlation_i = max(0, correlation_map[currency_pair[0:3]][i])
                correlation_j = max(0, correlation_map[currency_pair[4:7]][j])

                if correlation_i + correlation_j > 0:
                    matrix[i][j] = ((delta_i * correlation_i) + (delta_j * correlation_j)) / (correlation_i + correlation_j)
                    diffs.append(matrix[i][j])
                    diff_left.append(delta_i)
                    diff_right.append(delta_j) 

        diffs = np.mean(diffs)

        delta_final_map[currency_pair] = diffs


    return delta_final_map

def get_time_series_other(symbol, time, granularity="H1"):

    response_buffer = StringIO()
    curl = pycurl.Curl()

    curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=" + granularity + "&alignmentTimezone=America%2FNew_York")

    curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 8bde4f67a710b42553a821bdfff8efa9-eb1cb834f4060df9949504beb3356265'])

    curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

    curl.perform()
    curl.close()

    response_value = response_buffer.getvalue()
    j = json.loads(response_value)['candles']

    open_prices = []
    close_prices = []
    volumes = []
    times = []

    index = 0
    while index < len(j):
        item = j[index]

        s = item['time']
        s = s[0 : s.index('.')]
        timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

        times.append(timestamp)
        open_prices.append(item['openMid'])
        close_prices.append(item['closeMid'])
        volumes.append(item['volume'])
        index += 1

    return open_prices, close_prices, volumes, times

def create_correlation_graph_obv():

    item_ranking = {}
    vol_dir = {}
    for pair in currency_pairs:
        open_prices, close_prices, volumes, times = get_time_series_other(pair, 24 * 20 * 6,  granularity="H1")
        item_ranking[pair] = []
        vol_dir[pair] = []

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
                item_ranking[pair].append(z_score_price - z_score_obv)
                vol_dir[pair].append(z_score_obv)

    delta_map = {}
    for pair in currency_pairs:

        final_dir = np.percentile(vol_dir[pair], 50)

        if (vol_dir[pair] > 0) == (final_dir > 0):
            delta_map[pair] = np.percentile(item_ranking[pair], 50)
        else:
            delta_map[pair] = 0

    return delta_map

def create_correlation_graph_obv1():

    item_ranking = {}
    vol_dir = {}
    delta_map = {}
    for pair in currency_pairs:
        open_prices, close_prices, volumes, times = get_time_series_other(pair, 20 * 1,  granularity="D")
        item_ranking[pair] = []
        vol_dir[pair] = []


        obv = 0
        price = 0
        price_deltas = []
        obv_deltas = []
        for o, c, v, t in zip(open_prices, close_prices, volumes, times):

            if c > o:
                obv += v
                price += abs(c - o)
            else:
                obv -= v
                price -= abs(c - o)

            obv_deltas.append(obv)
            price_deltas.append(c)

        delta_map[pair] = np.corrcoef(obv_deltas, price_deltas)[0][1]

    return delta_map

delta_map1 = create_correlation_graph_obv1()
#delta_map2 = create_correlation_graph()

for pair in delta_map1:
    if abs(delta_map1[pair]) == 0:
        continue

    print (pair, delta_map1[pair])
   