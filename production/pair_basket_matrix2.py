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

        prices, times, volumes, lows, highs = load_time_series(compare_pair, None, False)
        before_price_df2 = pd.DataFrame()
        before_price_df2["prices" + str(i)] = prices
        before_price_df2["times"] = times
        before_price_df2["lows" + str(i)] = lows
        before_price_df2["highs" + str(i)] = highs
        count += 1

        if count > 1:
            before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
            before_all_price_df.reset_index(inplace=True)
        else:
            before_all_price_df = before_price_df2

    before_all_price_df.sort_values(by='times', ascending=True, inplace=True)

    times = before_all_price_df["times"].values.tolist()

    orders =[]
    min_profit_pips = []
    max_pip_loss_info = []
    equity = 0
    total_pnl = 0

    delta_window_max = []
    delta_window_min = []
    days_back_range = 20
    for index in range(24 * 20 * 8, len(before_all_price_df), 24):

        sorted_pairs = []
        open_price_map = {}
        open_price_low_map = {}
        open_price_high_map = {}
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

                    pair2 = before_all_price_df["prices" + str(j)].values.tolist()[-hours_back+index:index]
                    open_price_map[currency_pair] = pair2[-1]
                    open_price_low_map[currency_pair] = before_all_price_df["lows" + str(j)].tail(1).values.tolist()[-1]
                    open_price_high_map[currency_pair] = before_all_price_df["highs" + str(j)].tail(1).values.tolist()[-1]

                    if currency_pair[0:3] != select_currency:
                        pair2 = [1.0 / price for price in pair2]

                    correlations = []
                    for i, compare_pair in enumerate(currency_pairs):
                        if select_currency not in compare_pair:
                            continue

                        pair1 = before_all_price_df["prices" + str(i)].values.tolist()[-hours_back+index:index]
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

            diffs_long = []
            diffs_short = []

            for i in range(2, days_back_range):
                for j in range(2, days_back_range):

                    delta_i = delta_map[currency_pair[0:3]][i]
                    delta_j = delta_map[currency_pair[4:7]][j]

                    correlation_i = max(0, correlation_map[currency_pair[0:3]][i])
                    correlation_j = max(0, correlation_map[currency_pair[4:7]][j])

                    if correlation_i + correlation_j > 0:
                        matrix[i][j] = ((delta_i * correlation_i) + (delta_j * correlation_j)) / (correlation_i + correlation_j)
                        diffs_long.append(matrix[i][j])

                        if i < 11 and j < 11:
                            diffs_short.append(matrix[i][j])

            sorted_pairs.append([currency_pair, np.mean(diffs_long), np.mean(diffs_short)])

        sorted_pairs = sorted(sorted_pairs, key=lambda x: x[1], reverse=False)

        delta_window_min.append(abs(sorted_pairs[0][1]))
        delta_window_max.append(abs(sorted_pairs[-1][1]))

        if len(delta_window_max) > 5:
            delta_window_max = delta_window_max[1:]

        if len(delta_window_min) > 5:
            delta_window_min = delta_window_min[1:]
        

        pair_delta = {}
        pair_rank = {}
        count = 0
        for item in sorted_pairs:

            pair_delta[item[0]] = item[1]
            pair_rank[item[0]] = len(pair_rank)

            if item[1] >= max(delta_window_max):
                order = Order()

                total_orders = len([order.pair for order in orders if order.pair == item[0]])

                order.pair = item[0]
                order.dir = False
                order.open_price = open_price_map[order.pair]
                order.amount = abs(item[1]) 
                order.is_max = True
                order.max_pips = 0
                
                orders.append(order)

            if item[1] <= min(delta_window_min):
                order = Order()

                total_orders = len([order.pair for order in orders if order.pair == item[0]])

                order.pair = item[0]
                order.dir = True
                order.open_price = open_price_map[order.pair]
                order.amount = abs(item[1]) 
                order.is_max = False
                order.max_pips = 0
                
                orders.append(order)

        new_orders = []
        float_profit = 0
        for order in orders:

            if (open_price_map[order.pair] > order.open_price) == order.dir:
                profit = abs(open_price_map[order.pair] - order.open_price)
            else:
                profit = -abs(open_price_map[order.pair] - order.open_price)

            pip_size = 0.0001
            if "JPY" in order.pair:
                pip_size = 0.01

            profit /= pip_size
            profit -= 5

            
            if (open_price_low_map[order.pair] > order.open_price) == order.dir:
                profit_low = abs(open_price_low_map[order.pair] - order.open_price)
            else:
                profit_low = -abs(open_price_low_map[order.pair] - order.open_price)

            if (open_price_high_map[order.pair] > order.open_price) == order.dir:
                profit_high = abs(open_price_high_map[order.pair] - order.open_price)
            else:
                profit_high = -abs(open_price_high_map[order.pair] - order.open_price)

            profit_low /= pip_size
            profit_high /= pip_size

            min_profit = min(profit_high, profit_low)
            min_profit_pips.append(min_profit)
            max_pip_loss_info.append([min_profit, order.amount])

            '''
            if min(profit_high, profit_low) < order.max_pips - 50:
                profit = order.max_pips - 50
                profit *= order.amount
                equity += profit
                continue
            '''

            #order.max_pips = max([order.max_pips, profit_high, profit_low])

            profit *= order.amount

            if order.dir != (pair_delta[order.pair] < 0):
                equity += profit
                continue

            if order.is_max == True and (pair_rank[order.pair] < 27 - 8):
                equity += profit
                continue

            if order.is_max == False and (pair_rank[order.pair] > 8):
                equity += profit
                continue

            float_profit += profit
            new_orders.append(order)

        orders = new_orders


        if len(orders) > 0 and (float_profit + equity) / equity > 1 + (0.05 / len(orders)):
            equity += float_profit
            orders = []
            float_profit = 0


        print ("pnl", equity + float_profit, min(min_profit_pips), np.percentile(min_profit_pips, 10))

        pickle.dump(max_pip_loss_info, open("max_pip_loss_info.pickle", "wb"))
        #print (item[0], item[1], item[0][0:3] + ":" + str(item[2]), item[0][4:7] + ":" + str(item[3]))


create_correlation_graph()




