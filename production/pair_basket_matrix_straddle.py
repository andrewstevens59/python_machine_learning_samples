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

    orders = []
    equity = 0
    total_pnl = 0

    delta_window = []
    days_back_range = 20
    base_currency = "CAD"

    X_train = []
    for index in range(30 * 24, 24 * 250 * 6, 24):

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
                    open_price_low_map[currency_pair] = before_all_price_df["lows" + str(j)].values.tolist()[index]
                    open_price_high_map[currency_pair] = before_all_price_df["highs" + str(j)].values.tolist()[index]

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

            sorted_pairs.append([currency_pair, diffs, matrix])

        sorted_pairs = sorted(sorted_pairs, key=lambda x: abs(x[1]), reverse=True)

        X_train.append([sorted_pairs, open_price_map, open_price_low_map, open_price_high_map])

        pickle.dump(X_train, open("stat_arb_train_second_test_set.pickle", "wb"))

def replay_data(base_currency, X_train, sl_pips, tp_pips):

    profits = []
    orders = []
    equity = 5000
    total_pnl = 0

    delta_window = []
    draw_downs = []
    max_orders = 0
    max_equity = 0

    total_pnl_series = 0
    hold_times = []

    pnl_window = []

    for index in range(len(X_train)):

        x = X_train[index]
        sorted_pairs = x[0]
        open_price_map = x[1]
        open_price_low_map = x[2]
        open_price_high_map = x[3]

        new_sorted_pairs = []
        for item in sorted_pairs:
            matrix = item[2]

            diffs = []
            for i in range(2, 20):
                for j in range(2, 20):

                    if matrix[i][j] != 0:
                        diffs.append(matrix[i][j])

            item[1] = np.mean(diffs)

            new_sorted_pairs.append(item)

        sorted_pairs = new_sorted_pairs
        sorted_pairs = sorted(sorted_pairs, key = lambda x: abs(x[1]), reverse=True)
        global_pairs = sorted_pairs

        
        if base_currency != "ALL":
            sorted_pairs = [v for v in sorted_pairs if base_currency in v[0]]
        

        delta_window.append(abs(sorted_pairs[0][1]))

        max_pair = sorted_pairs[0][0]

        if len(delta_window) > 5:
            delta_window = delta_window[1:]

        if len(orders) == 0:
            start_equity = equity
            total_pnl_series = 0 

        max_pip_profit = 0
        if len(orders) > 0:
            max_pip_profit = max([order.pip_profit for order in orders])


        pair_delta = {}
        pair_rank = {}
        global_rank = {}
        count = 0
        for item in sorted_pairs:

            pair_delta[item[0]] = item[1]
            pair_rank[item[0]] = len(pair_rank)
            global_rank[item[0]] = [i for i, v in enumerate(global_pairs) if v[0] == item[0]][0]

            mean = np.mean(delta_window)
            std = np.mean(delta_window)

            z_score = (item[1] - mean) / max(0.055, std)

            if abs(item[1]) > 1:
                

                #total_orders = len([order.pair for order in orders if order.pair == item[0]])

                pip_size = 0.0001
                if "JPY" in item[0]:
                    pip_size = 0.01

                order = Order()
                order.pair = item[0]
                order.dir = item[1] < 0
                order.is_revert = False

                order.open_price = open_price_map[order.pair]
                order.amount = 1
                order.open_delta = abs(item[1]) 
                order.max_pips = 0
                order.min_profit = 0
                order.pip_profit = 0
                order.count = 1
                order.open_time = index
                orders.append(order)

                order = Order()
                order.pair = item[0]
                order.dir = item[1] > 0
                order.is_revert = False

                order.open_price = open_price_map[order.pair]
                order.amount = 1
                order.open_delta = abs(item[1]) 
                order.max_pips = 0
                order.min_profit = 0
                order.pip_profit = 0
                order.count = 1
                order.open_time = index
                orders.append(order)

  

        new_orders = []
        float_profit = 0
        total_pips = 0

        existing_order_count = 0
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
            order.min_profit = min(order.min_profit, min_profit)
            order.pip_profit = min(profit_high, profit_low)

            '''
            if order.min_profit < -150 and profit > -150:
                equity += (profit - 150) * order.amount
                print ("stop loss ", (profit - 150))
                order.min_profit = 0

            if profit < -150:
                print ("cap hedge profit")
                profit = -150
            '''

            
            if min(profit_high, profit_low) < -50:
                print ("stop")

                #total_orders = len([order.pair for order in orders if order.pair == item[0]])

                pip_size = 0.0001
                if "JPY" in item[0]:
                    pip_size = 0.01

                new_order = Order()
                new_order.pair = order.pair
                new_order.dir = (order.dir == False)
                new_order.is_revert = False
                new_order.open_price = open_price_map[order.pair]
                new_order.amount = order.amount * 1
                new_order.max_pips = 0
                new_order.min_profit = 0
                new_order.pip_profit = 0
                new_order.count = order.count * 2
                new_order.open_time = index
                new_orders.append(new_order)

                new_order = Order()
                new_order.pair = order.pair
                new_order.dir = (order.dir == True)
                new_order.is_revert = False
                new_order.open_price = open_price_map[order.pair]
                new_order.amount = order.amount * 1
                new_order.max_pips = 0
                new_order.min_profit = 0
                new_order.pip_profit = 0
                new_order.count = order.count * 2
                new_order.open_time = index
                new_orders.append(new_order)

                equity += -50 * order.amount 
                continue

  
            '''
            if max(profit_high, profit_low) > order.take_profit:
                print ("stop")

                equity += order.take_profit * order.amount
                continue
            '''
            
            '''
            if min(profit_high, profit_low) < -300:
                print ("stop")

                equity += -300 * order.amount
                hold_times.append(index - order.open_time)
                continue
            '''
            
            '''
            if order.dir:
                order.stop_loss = max(order.stop_loss, abs(order.open_price - (open_price_map[order.pair] - (300 * pip_size))) / pip_size)
            else:
                order.stop_loss = max(order.stop_loss, abs(order.open_price - (open_price_map[order.pair] + (300 * pip_size))) / pip_size)
            
            if order.dir:
                order.take_profit = abs(order.open_price - (open_price_map[order.pair] + (100 * pip_size))) / pip_size
            else:
                order.take_profit = abs(order.open_price - (open_price_map[order.pair] - (100 * pip_size))) / pip_size
            '''

            total_pips += profit 

            #order.max_pips = max([order.max_pips, profit_high, profit_low])

            profit *= order.amount 
            
            '''
            if profit < 0:
                if order.is_revert == False:

                    if pair_rank[order.pair] > 5:
                        hold_times.append(index - order.open_time)
                        equity += profit
                        continue

                    if order.dir != (pair_delta[order.pair] < 0):
                        hold_times.append(index - order.open_time)
                        equity += profit
                        continue
                else:

                    if pair_rank[order.pair] > 5:
                        hold_times.append(index - order.open_time)
                        equity += profit
                        continue

                    if order.dir != (pair_delta[order.pair] > 0):
                        hold_times.append(index - order.open_time)
                        equity += profit
                        continue

            if profit > 0:
                if pair_rank[order.pair] > 3:# or global_rank[order.pair] > 3:
                    hold_times.append(index - order.open_time)
                    equity += profit
                    continue
            '''

            '''
            if global_rank[order.pair] > 14:
                hold_times.append(index - order.open_time)
                equity += profit
                continue
            '''

            float_profit += profit
            new_orders.append(order)
            existing_order_count += order.count

        if len(orders) > 0:
            max_orders = max(max_orders, max([order.count for order in orders]))

        orders = new_orders
        print ("orders", len(orders))
        
        if total_pips >= 400 and float_profit > 0:
            orders = new_orders
            equity += float_profit 
            orders = []
            float_profit = 0
        

        #print ("pnl", base_currency, equity + float_profit)

        profits.append(equity + float_profit)
        print (equity + float_profit)

        max_equity = max(max_equity, equity)
        draw_downs.append((equity) / max_equity)

        #print (item[0], item[1], item[0][0:3] + ":" + str(item[2]), item[0][4:7] + ":" + str(item[3]))

        
    return profits, hold_times

#create_correlation_graph()

sharpes = []
 
def best_fit_curve():

    best_fit_map = {}
    X_train1 = pickle.load(open("stat_arb_train_second_test_set.pickle", "rb"))
    X_train2 = pickle.load(open("stat_arb_train_second1.pickle", "rb"))
    X_train = X_train1 + X_train2

    profit_curves = []
    total_equity = 0
    for base_currency in ["EUR", "USD", "AUD", "CAD", "JPY", "NZD", "CHF", "GBP"]:

        best_tp_pips = 0
        best_sl_pips = 0
        best_sharpe = 0

        for tp_pips in range(50, 300, 20):

            for sl_pips in range(50, 100, 10):

                profits, hold_times = replay_data(base_currency, X_train, tp_pips, sl_pips)

                returns = [b - a for b, a in zip(profits[1:], profits[:-1]) if abs(b - a) > 0 ]
                sharpe = np.mean(returns) / np.std(returns)

                if sharpe > best_sharpe:
                    best_sl_pips = sl_pips
                    best_tp_pips = tp_pips
                    best_sharpe = sharpe

                print (base_currency, sharpe, sl_pips, tp_pips)

        best_fit_map[base_currency] = {"sl_pips" : best_sl_pips, "tp_pips" : best_tp_pips}

    print (best_fit_map)


def show_fitted_params():

    '''
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

    base_time = times[0]
    '''

    X_train1 = pickle.load(open("stat_arb_train_second_test_set.pickle", "rb"))
    X_train2 = pickle.load(open("stat_arb_train_second1.pickle", "rb"))
    X_train = X_train1 + X_train2
    params = {'USD': {'sl_pips': 90, 'tp_pips': 50}, 'AUD': {'sl_pips': 70, 'tp_pips': 50}, 'CHF': {'sl_pips': 50, 'tp_pips': 110}, 'JPY': {'sl_pips': 60, 'tp_pips': 70}, 'GBP': {'sl_pips': 50, 'tp_pips': 70}, 'NZD': {'sl_pips': 90, 'tp_pips': 50}, 'EUR': {'sl_pips': 60, 'tp_pips': 70}, 'CAD': {'sl_pips': 50, 'tp_pips': 50}}

    profit_curves = []
    total_equity = 0
    all_returns = []
    all_hold_times = []
    for base_currency in ["ALL"]:#["AUD", "NZD", "GBP", "CHF", "EUR", "USD", "JPY", "CAD"]:

        profits, hold_times  = replay_data(base_currency, X_train, params["GBP"]["tp_pips"], min(80, params["GBP"]["sl_pips"]))

        all_returns += [b - a for a, b in zip(profits[:-1], profits[1:]) if abs(a - b) > 0]
        profit_curves.append(profits)
        all_hold_times += hold_times

    '''
    for percentile in range(10, 100, 10):
        print ("hold times", percentile, np.percentile(all_hold_times, percentile))
    '''

    print ("mean sharpe", np.mean(all_returns) / np.std(all_returns))


    import matplotlib.pyplot as plt
    index = 0
    for base_currency in ["ALL"]:#["AUD", "NZD", "GBP", "CHF", "EUR", "USD", "JPY", "CAD"]:

        plt.plot(range(len(profit_curves[index])), profit_curves[index], label=base_currency + " Pairs")
        index += 1

    '''
    x_tick_indexes = range(0, len(profit_curves[0]), 150)
    from datetime import datetime, timedelta 
    base = datetime.fromtimestamp(base_time)

    date_list = [(base + timedelta(days=x+(6 * 250))).strftime('%m-%Y') for x in x_tick_indexes]
    plt.xticks(x_tick_indexes, date_list, rotation=30)
    '''
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Stat Arb Strategy - Available On QuantForexTrading.com")
    plt.legend()
    plt.show()

def get_time_series(symbol, time, granularity="H1"):

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


show_fitted_params()
#best_fit_curve()
#sys.exit(0)
#create_correlation_graph()



