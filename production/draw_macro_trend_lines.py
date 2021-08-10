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

import matplotlib
matplotlib.use('Agg')

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
from bisect import bisect

import paramiko
import json

import logging
import os
import enum

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

checkIfProcessRunning('draw_macro_trend_lines.py', "")

if get_mac() != 150538578859218:
    root_dir = "/root/trading/production/" 
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

    index = 0
    while index < len(j):
        item = j[index]

        s = item['time']
        s = s[0 : s.index('.')]
        timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

        if item['lowMid'] != item['highMid'] or item['volume'] > 0:
            times.append(timestamp)
            prices.append(item['closeMid'])
            index += 1

    return prices, times


def check_memory():
    import psutil
    import gc

    memory = psutil.virtual_memory() 
    while memory.percent > 80:
        gc.collect()
        memory = psutil.virtual_memory() 

# solve for a and b
def best_fit(X, Y):

    b, a = np.polyfit(X, Y, 1)

    return a, b

def store_model_prediction(pair, levels, curr_price):

    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    cursor = cnx.cursor()


    query = ("""INSERT INTO signal_summary(timestamp, pair, model_group, forecast_percentiles) 
                values (now(),'{}','{}','{}')""".
        format(
            pair,
            "Support And Resistance",
            json.dumps({"levels" : levels, "curr_price" : curr_price})
            ))

    print (query)

    cursor.execute(query)
    cnx.commit()

def store_straddle_suggestion(pair):

    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    cursor = cnx.cursor()


    query = ("""INSERT INTO signal_summary(timestamp, pair, model_group) 
                values (now(),'{}','{}')""".
        format(
            pair,
            "Straddle Recommend",
            ))

    print (query)

    cursor.execute(query)
    cnx.commit()

def get_future_movement(a, b, time_offset, future_prices):


    future_time_periods = [1, 2, 3, 4, 5, 10, 15, 20]
    future_time_periods = [t for t in future_time_periods]

    for period in future_time_periods:
        yfit = a + b * (time_offset + period)

        if period not in future_prices:
            future_prices[period] = []

        future_prices[period].append(yfit)

    time_periods = [t for t in range(-365, 20)]

    trend_lines = []
    for period in time_periods:
        yfit = a + b * (period)
        trend_lines.append(yfit)

    return trend_lines


def calculate_pivot_score(curr_price, prices, pip_size):

    scores = []
    for price1 in prices:
        diffs = []
        for price2 in prices:
            if abs(price1 - price2) / pip_size > 10:
                diffs.append(1.0 / abs(price1 - price2))

        scores.append(np.mean(diffs))

    diff = []
    for price2 in prices:
        if abs(curr_price - price2) / pip_size > 10:
            diffs.append(1.0 / abs(curr_price - price2))

    scores = sorted(scores)

    index = bisect(scores, np.mean(diffs))
    percentile = 100 - int((float(index) / len(scores)) * 100)

    return percentile


def get_today_prediction():

    for pair in currency_pairs:

        if pair[4:7] == "JPY":
            pip_size = 0.01
        else:
            pip_size = 0.0001

        prices, times = get_time_series(pair, 400, granularity="D")

        lines = []

        import matplotlib.pyplot as plt

        min_prices = min(prices)
        max_prices = max(prices)
        pip_range = (max(prices) - min(prices)) 
        pip_range /= 2

        X_set = []
        y_set = []

        future_prices = {}
        future_trends = []
        future_time_periods = [1, 2, 3, 4, 5, 10, 15, 20]
        for look_back in range(20, len(prices), 20):
            Y = prices[len(prices)-look_back:]
            X = [len(prices) - look_back + x for x in range(len(Y))]

            # solution
            a, b = best_fit(X, Y)

            plt.scatter(X, Y)
            yfit = [a + b * xi for xi in X]


            if abs(yfit[-1] - Y[-1]) < pip_range:
                X_set.append(X)
                y_set.append(yfit)
                future_trends.append(get_future_movement(a, b, len(prices), future_prices))

            for percentile in [20, 30, 70, 80]:

                threshold = np.percentile(Y, percentile)

                if percentile > 50:
                    series = [[x, y] for x, y in zip(X, Y) if y > threshold]
                else:
                    series = [[x, y] for x, y in zip(X, Y) if y < threshold]

                x = [a[0] for a in series]
                y = [a[1] for a in series]

                if len(x) < 10:
                    continue

                # solution
                a, b = best_fit(x, y)
     
                plt.scatter(x, y)
                yfit = [a + b * xi for xi in X]

                if abs(yfit[-1] - Y[-1]) < pip_range:
                    X_set.append(X)
                    y_set.append(yfit)
                    future_trends.append(get_future_movement(a, b, len(prices), future_prices))


        plt.figure(figsize=(8,4))
        plt.title(pair + " Support And Resistance Trend Lines")

        y_ends = []
        for x, y in zip(X_set, y_set):
            y_ends.append(y[-1])

            x_fin = [x1 for x1, y1 in zip(x, y) if abs(y1 - prices[-1]) < pip_range or (y1 > min_prices and y1 < max_prices)]
            y_fin = [y1 for x1, y1 in zip(x, y) if abs(y1 - prices[-1]) < pip_range or (y1 > min_prices and y1 < max_prices)]
            plt.plot(x_fin, y_fin, alpha=0.5)

        y_ends = sorted(y_ends)
        index = bisect(y_ends, prices[-1])

        percentile = int((float(index) / len(y_ends)) * 100)

        plt.plot(range(len(prices)), prices, color='black')
        plt.xticks(range(0, len(prices), 20), [datetime.datetime.utcfromtimestamp(times[t]).strftime('%y-%m') for t in range(0, len(prices), 20)], rotation=30)
        plt.ylabel("Price")
     
        plt.savefig("/var/www/html/images/{}_support_resistance.png".format(pair))
        plt.close()

        plt.figure(figsize=(8,5))
        plt.title(pair + " Straddle Trade Setup")
        plt.axvline(0, color='black')
        plt.ylabel("Price")

        kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

        time_periods = [t + len(prices) for t in range(-365, 20)]
        time_indexes = [len(prices) + t for t in range(-365, 0)]

        plt.xlim(time_periods[0],time_periods[-1])

        resistances = []
        supports = []
        for index in range(len(future_trends)):
            y_end = future_trends[index][len(time_indexes)-1]
    
            if y_end > prices[-1]:
                resistances.append([future_trends[index], y_end])
            else:
                supports.append([future_trends[index], y_end])

        resistances = sorted(resistances, key=lambda x: x[1])
        supports = sorted(supports, key=lambda x: x[1], reverse=True)

        if pair[4:7] == "JPY":
            decimals = 2
        else:
            decimals = 4

        for percentile in [5, 10, 15, 20]:

            if len(supports) > 0:
                support_vals = supports[int(len(supports) * percentile * 0.01)]
                pip_diff = int(abs(support_vals[1] - prices[-1]) / pip_size)
                plt.plot(time_periods, support_vals[0], color='red', label="LIMIT SELL {}, {} pips ({}%)".format(round(support_vals[1], decimals), pip_diff, percentile))

            if len(resistances) > 0:
                resistance_vals = resistances[int(len(resistances) * percentile * 0.01)]
                pip_diff = int(abs(resistance_vals[1] - prices[-1]) / pip_size)
                plt.plot(time_periods, resistance_vals[0], color='lime', label="LIMIT BUY {}, {} pips ({}%)".format(round(resistance_vals[1], decimals), pip_diff, percentile))

            if percentile == 5:
                if len(supports) > 0 and len(resistances) > 0 and abs(support_vals[1] - resistance_vals[1]) < 60 * pip_size:
                    store_straddle_suggestion(pair)

        plt.plot(time_indexes, prices[-365:], color='black')

        x_tick_indexes = [time_indexes[index] for index in range(0, len(time_indexes), 20)]
        plt.xticks(x_tick_indexes, [datetime.datetime.utcfromtimestamp(times[t]).strftime('%y-%m') for t in x_tick_indexes], rotation=30)
        plt.ylabel("Price")
        plt.legend()
        plt.savefig("/var/www/html/images/{}_straddle_trade.png".format(pair))
        plt.close()

        plt.figure(figsize=(8,5))
        plt.title(pair + " Support And Resistance Price Levels")
        plt.axvline(0, color='black')
        plt.ylabel("Probability")

        plt.xlabel("Support <-- Pips From Current Price --> Resistance")

        kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

        dists = []
        price_levels = []
      
        for time_frame, future_period in zip(["1 Day", "2 Days", "3 Days", "4 Days", "1 Week", "2 Weeks", "3 Weeks", "1 Month"], [1, 2, 3, 4, 5, 10, 15, 20]):
            sns.distplot([(y - prices[-1]) / pip_size for y in future_prices[future_period]], label=time_frame + " Forecast", **kwargs)
            dists += [(y - prices[-1]) / pip_size for y in future_prices[future_period]]
            price_levels += future_prices[future_period]

        store_model_prediction(pair, price_levels, prices[-1])

   
        chosen_percentiles = [5, 25, 50, 75]

        for index in range(len(chosen_percentiles)):

            percentile = 100 - chosen_percentiles[index]

            supports = [v for v in dists if v < 0]
            line = int(np.percentile(supports, percentile))

            supports = [v for v in price_levels if v < prices[-1]]
            price_level = round(np.percentile(supports, percentile), decimals)

            plt.axvline(line, color='red', label="{}% ({} / {} pips) Support".format(chosen_percentiles[index], price_level, line))

            percentile = chosen_percentiles[index]

            resistances = [v for v in dists if v > 0]
            line = int(np.percentile(resistances, percentile))

            resistances = [v for v in price_levels if v > prices[-1]]
            price_level = round(np.percentile(resistances, percentile), decimals)

            plt.axvline(line, color='lime', label="{}% ({} / {} pips) Resistance".format(chosen_percentiles[index], price_level, line))

        plt.legend()
        plt.savefig("/var/www/html/images/{}_support_resistance_distribution.png".format(pair))
        plt.close()

        plt.figure(figsize=(8,5))
        plt.title(pair + " Trade Entries")

        chosen_percentiles = range(10, 80, 10)
        for index in range(len(chosen_percentiles)):

            if index < len(chosen_percentiles) - 1:
                style = '--'
            else:
                style = '-'

            percentile = 100 - chosen_percentiles[index]
            supports = [v for v in dists if v < 0]
            pips = int(np.percentile(supports, percentile))

            supports = [v for v in price_levels if v < prices[-1]]
            price_level = round(np.percentile(supports, percentile), 4)

            plt.axhline(price_level, linestyle=style, color='red', label="{}% ({} / {} pips) SELL".format(chosen_percentiles[index], price_level, pips))

            percentile = chosen_percentiles[index]

            resistances = [v for v in dists if v > 0]
            pips = int(np.percentile(resistances, percentile))

            resistances = [v for v in price_levels if v > prices[-1]]
            price_level = round(np.percentile(resistances, percentile), 4)

            plt.axhline(price_level, linestyle=style, color='lime', label="{}% ({} / {} pips) BUY".format(chosen_percentiles[index], price_level, pips))


        plt.plot(time_indexes, prices[-365:], color='black')

        x_tick_indexes = [time_indexes[index] for index in range(0, len(time_indexes), 20)]
        plt.xticks(x_tick_indexes, [datetime.datetime.utcfromtimestamp(times[t]).strftime('%y-%m') for t in x_tick_indexes], rotation=30)
        plt.ylabel("Price")

        plt.legend()
        plt.savefig("/var/www/html/images/{}_trade_entries.png".format(pair))
        plt.close()
   
trade_logger = setup_logger('first_logger', root_dir + "trend_lines.log")
trade_logger.info('Starting ') 


try:
    get_today_prediction()
    trade_logger.info('Finished ') 
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())

