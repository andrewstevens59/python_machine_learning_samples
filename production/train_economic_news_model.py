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
from bisect import bisect

import paramiko
import json

import logging
import os
import enum

import matplotlib
matplotlib.use('Agg')

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from bisect import bisect
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

checkIfProcessRunning('train_economic_news_model.py', "")

if get_mac() != 150538578859218:
    root_dir = "/root/trading/production/" 
else:
    root_dir = "" 

select_day_ranges = [2, 3, 4, 6, 8, 10, 15, 20, 25, 30]
barrier_levels = [10, 20, 30, 40, 50, 100, 150, 200, 250] 

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

def calculate_time_diff(now_time, ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    date = datetime.datetime.utcfromtimestamp(now_time).strftime('%Y-%m-%d %H:%M:%S')
    e = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    _diff = (e - s)

    while s < e:
        max_hour = 24
        if s.day == e.day:
            max_hour = e.hour

        if s.weekday() in {4}:
            max_hour = 21

        if s.weekday() in {4} and s.hour in {21, 22, 23}:
            hours = 1
            _diff -= timedelta(hours=hours)
        elif s.weekday() in {5}:
            hours = max_hour - s.hour
            _diff -= timedelta(hours=hours)
        elif s.weekday() in {6} and s.hour < 21:
            hours = min(21, max_hour) - s.hour
            _diff -= timedelta(hours=hours)
        else:
            hours = max_hour - s.hour

        if hours == 0:
            break
        s += timedelta(hours=hours)

    return (_diff.total_seconds() / (60 * 60))



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
                prices.append(c_price)
                times.append(time)
                volumes.append(volume)

    return prices, times, volumes

def get_calendar_df(pair, year): 

    if pair != None:
        currencies = [pair[0:3], pair[4:7]]
    else:
        currencies = None

    if get_mac() == 150538578859218:
        with open("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt") as f:
            content = f.readlines()
    else:
        with open("/root/trading_data/calendar_" + str(year) + ".txt") as f:
            content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    lines = [x.strip() for x in content] 

    from_zone = tz.gettz('US/Eastern')
    to_zone = tz.gettz('UTC')

    contents = []

    for line in lines:
        line = line[len("2018-12-23 22:44:55 "):]
        toks = line.split(",")

        if currencies == None or toks[1] in currencies:

            time = int(toks[0])
            actual = unicode(toks[3], 'utf-8')
            forecast = unicode(toks[4], 'utf-8')
            previous = unicode(toks[5], 'utf-8')

            actual = "".join([v for v in actual if v.isnumeric() or v in ['.', '+', '-']])
            if len(actual) == 0:
                continue

            try:
                actual = float(actual)

                forecast = "".join([v for v in forecast if v.isnumeric() or v in ['.', '+', '-']])
                if len(forecast) > 0:
                    forecast = float(forecast)
                else:
                    forecast = actual

                previous = "".join([v for v in previous if v.isnumeric() or v in ['.', '+', '-']])
                if len(previous) > 0:
                    previous = float(previous)
                else:
                    previous = actual

                contents.append([toks[1], time, toks[2], actual, forecast, previous, int(toks[6]), toks[7]])
            except:
                pass

    return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "impact", "better_worse"])

if get_mac() == 150538578859218:
    news_release_stat_df = pd.read_csv("{}../news_dist_stats.csv".format(root_dir))
else:
    news_release_stat_df = pd.read_csv("{}../../news_dist_stats.csv".format(root_dir))

news_release_stat_df.set_index("key", inplace=True)

def news_movement_feature_ranking(curr_time, curr_calendar_df, max_time_lag, before_prices):

    X_last = []
    z_score_dists1 = {}
    z_score_dists2 = {}

    for index, row in curr_calendar_df.iterrows():

        time_lag = calculate_time_diff(curr_time, row["time"])

        if time_lag > max_time_lag:
            continue

        key = row["description"] + "_" + row["currency"]
        stat_row = news_release_stat_df[news_release_stat_df.index == key]
        if len(stat_row) == 0:
            continue

        stat_row = stat_row.iloc[0]

        sign = stat_row["sign"]

        if stat_row["forecast_std"] > 0:
            z_score1 = (float(row["actual"] - row["forecast"]) - stat_row["forecast_mean"]) / stat_row["forecast_std"]
        else:
            z_score1 = None

        if stat_row["previous_std"] > 0:
            z_score2 = (float(row["actual"] - row["previous"]) - stat_row["previous_mean"]) / stat_row["previous_std"]
        else:
            z_score2 = None

        time_lag = (curr_time - row["time"]) / (60 * 60) 
        currency = row["currency"]
        impact = row["impact"]
        

        if row["actual"] > row["forecast"]:
            diff1 = sign
        elif row["actual"] < row["forecast"]:
            diff1 = -sign
        else:
            diff1 = 0

        if row["actual"] > row["previous"]:
            diff2 = sign
        elif row["actual"] < row["previous"]:
            diff2 = -sign
        else:
            diff2 = 0

        if currency not in z_score_dists1:
            z_score_dists1[currency] = {}
            z_score_dists2[currency] = {}

        for impact_index in [0, impact]:
            if impact_index not in z_score_dists1[currency]:
                z_score_dists1[currency][impact_index] = []
                z_score_dists2[currency][impact_index] = []


            if z_score1 != None:
                z_score_dists1[currency][impact_index].append(z_score1 * sign)

            if z_score2 != None:
                z_score_dists2[currency][impact_index].append(z_score2 * sign)

    for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:

        for impact in range(0, 4):

            if currency in z_score_dists1 and impact in z_score_dists1[currency]:
                dist1 = z_score_dists1[currency][impact]
                dist2 = z_score_dists2[currency][impact]

                if len(dist1) < 3:
                    dist1 += [0] * (3 - len(dist1))

                if len(dist2) < 3:
                    dist2 += [0] * (3 - len(dist2))

                dist1 = sorted(dist1, key=lambda x: abs(x), reverse=True)
                dist2 = sorted(dist2, key=lambda x: abs(x), reverse=True)

                X_last += dist1[:3]
                X_last += dist2[:3]
            else:
                X_last += [0] * 3
                X_last += [0] * 3

    if before_prices != None:
        for lag in [24, 48, 72, 96]:
            X_last.append(before_prices[-1] - before_prices[-lag])

    return X_last


def news_movement_feature_simple(curr_time, curr_calendar_df, max_time_lag, before_prices):


    currency_news1_count = {}
    currency_news2_count = {}
    currency_news3_count = {}
    
    z_score_dists1 = {}
    z_score_dists2 = {}
    for index, row in curr_calendar_df.iterrows():
        time_lag = calculate_time_diff(curr_time, row["time"])

        if time_lag > max_time_lag:
            continue

        key = row["description"] + "_" + row["currency"]
        stat_row = news_release_stat_df[news_release_stat_df.index == key]
        if len(stat_row) == 0:
            continue

        stat_row = stat_row.iloc[0]

        sign = stat_row["sign"]

        if stat_row["forecast_std"] > 0:
            z_score1 = (float(row["actual"] - row["forecast"]) - stat_row["forecast_mean"]) / stat_row["forecast_std"]
        else:
            z_score1 = None

        if stat_row["previous_std"] > 0:
            z_score2 = (float(row["actual"] - row["previous"]) - stat_row["previous_mean"]) / stat_row["previous_std"]
        else:
            z_score2 = None

        time_lag = (curr_time - row["time"]) / (60 * 60) 
        currency = row["currency"]
        impact = row["impact"]
        

        if row["actual"] > row["forecast"]:
            diff1 = sign
        elif row["actual"] < row["forecast"]:
            diff1 = -sign
        else:
            diff1 = 0

        if row["actual"] > row["previous"]:
            diff2 = sign
        elif row["actual"] < row["previous"]:
            diff2 = -sign
        else:
            diff2 = 0

        if currency not in currency_news1_count:
            currency_news1_count[currency] = {}
            currency_news2_count[currency] = {}
            currency_news3_count[currency] = {}
            z_score_dists1[currency] = {}
            z_score_dists2[currency] = {}

        for impact_index in [0, impact]:
            if impact_index not in currency_news1_count[currency]:
                currency_news1_count[currency][impact_index] = [0]
                currency_news2_count[currency][impact_index] = [0]
                currency_news3_count[currency][impact_index] = [0]
                z_score_dists1[currency][impact_index] = []
                z_score_dists2[currency][impact_index] = []

            currency_news1_count[currency][impact_index][0] += diff1
            currency_news2_count[currency][impact_index][0] += diff2
            currency_news3_count[currency][impact_index][0] += 1

            if z_score1 != None:
                z_score_dists1[currency][impact_index].append(z_score1 * sign)

            if z_score2 != None:
                z_score_dists2[currency][impact_index].append(z_score2 * sign)

    X_last = []
    for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:

        for impact in range(0, 4):
            if currency in currency_news1_count and impact in currency_news1_count[currency]:
                X_last += currency_news1_count[currency][impact]
                X_last += currency_news2_count[currency][impact]
                X_last += currency_news3_count[currency][impact]

                dist1 = z_score_dists1[currency][impact]
                dist2 = z_score_dists2[currency][impact]

                X_last += list(np.histogram(dist1, bins=[-2000, -2, -1, 0, 1, 2, 2000])[0])
                X_last += list(np.histogram(dist2, bins=[-2000, -2, -1, 0, 1, 2, 2000])[0])
            else:
                X_last += [0] * 1
                X_last += [0] * 1
                X_last += [0] * 1

                X_last += [0] * 6
                X_last += [0] * 6

    if before_prices != None:
        for lag in [24, 48, 72, 96]:
            X_last.append(before_prices[-1] - before_prices[-lag])

    return X_last


def train_regression_model(pair, calendar_df):
    prices, times, volumes = load_time_series(pair, None, False)

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times
    price_df.set_index("times", inplace=True)

    for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:

        models = []
        for train_offset in [0, 24 * 1, 24 * 2, 24 * 3]:

            for max_time_lag in [48, 96]:

                print ("train offset", train_offset, "max time lag", max_time_lag)

                index = train_offset
                X_train = []
                y_train = []
                while index < len(times):
                    curr_time = times[index]

                    curr_calendar_df = calendar_df[(calendar_df["time"] < curr_time) & (calendar_df["time"] > curr_time - (7 * 24 * 60 * 60))]

                    after_prices = (price_df[price_df.index >= curr_time])["prices"].values.tolist()
                    before_prices = (price_df[price_df.index <= curr_time]).tail(96)["prices"].values.tolist()

                    if len(after_prices) < 30 * 24 or len(before_prices) < 4 * 24:
                        index += 24 * 4
                        continue

                    if feature_type == "news_movement_feature_simple":
                        feature_vector = news_movement_feature_simple(curr_time, curr_calendar_df, max_time_lag, before_prices)
                    else:
                        feature_vector = news_movement_feature_ranking(curr_time, curr_calendar_df, max_time_lag, before_prices)

                    forecast = []
                    for day_range in select_day_ranges:
                        hour_range = day_range * 24
                        forecast.append(after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0])

                    X_train.append(feature_vector)
                    y_train.append(forecast)


                    index += 24 * 4


                for day_range in range(len(y_train[0])):
                    for percentile in [50]:
                        y = [y[day_range] for y in y_train]

                        print ("day_range" , day_range, len(y_train[0]))
                        cutoff = np.percentile([abs(y1) for y1 in y], percentile)
                        train = [[X, y1] for X, y1 in zip(X_train, y) if abs(y1) > cutoff]
                        X = [v[0] for v in train]
                        y = [1 if v[1] > 0 else -1 for v in train]

                        param_list = [
                            None,
                            {"learning_rate" : 0.8794, "n_estimators" : 703, "max_depth" : 3, "gamma" : 0.8492},
                        ]

                        for params in param_list:
                            

                            if params == None:
                                clf = xgb.XGBRegressor()
                            else:
                                clf = xgb.XGBRegressor(
                                    max_depth=int(round(params["max_depth"])),
                                    learning_rate=float(params["learning_rate"]),
                                    n_estimators=int(params["n_estimators"]),
                                    gamma=params["gamma"])

                            clf.fit(np.array(X), y)
                            predictions = clf.predict(np.array(X))

                        if clf != None:
                            models.append({"max_time_lag" : max_time_lag, "feature_type" : feature_type, "percentile" : percentile, "day_range" : day_range, "model" : clf, "predictions" : predictions})
                            

        pickle.dump(models, open("regression_economic_model_{}_{}.pickle.gz".format(pair, feature_type), "wb"))


def train_one_day_regression_model(pair, calendar_df):
    prices, times, volumes = load_time_series(pair, None, False)

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times

    for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:

        models = []
        for train_offset in [0, 24 * 1, 24 * 2, 24 * 3]:

            for max_time_lag in [6, 12, 24, 48, 96]:

                print ("train offset", train_offset, "max time lag", max_time_lag)

                index = train_offset
                X_train = []
                y_train = []
                while index < len(times):
                    curr_time = times[index]

                    curr_calendar_df = calendar_df[(calendar_df["time"] < curr_time) & (calendar_df["time"] > curr_time - (7 * 24 * 60 * 60))]

                    after_prices = (price_df[price_df["times"] >= curr_time])["prices"].values.tolist()

                    if len(after_prices) < 30 * 24:
                        index += 24 * 4
                        continue

                    if feature_type == "news_movement_feature_simple":
                        feature_vector = news_movement_feature_simple(curr_time, curr_calendar_df, max_time_lag, None)
                    else:
                        feature_vector = news_movement_feature_ranking(curr_time, curr_calendar_df, max_time_lag, None)

                    hour_range = 1 * 24
                    y_train.append(after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0])
                    X_train.append(feature_vector)

                    index += 24 * 4


                for percentile in [0]:
                    y = y_train

                    print ("percentile" , percentile)
                    cutoff = np.percentile([abs(y1) for y1 in y], percentile)
                    train = [[X, y1] for X, y1 in zip(X_train, y) if abs(y1) > cutoff]
                    X = [v[0] for v in train]
                    y = [1 if v[1] > 0 else -1 for v in train]

                    param_list = [
                        None,
                        {"learning_rate" : 0.8794, "n_estimators" : 703, "max_depth" : 3, "gamma" : 0.8492},
                    ]

                    for params in param_list:
                        
         
                        if params == None:
                            clf = xgb.XGBRegressor()
                        else:
                            clf = xgb.XGBRegressor(
                                max_depth=int(round(params["max_depth"])),
                                learning_rate=float(params["learning_rate"]),
                                n_estimators=int(params["n_estimators"]),
                                gamma=params["gamma"])

                        clf.fit(np.array(X), y)
                        predictions = clf.predict(np.array(X))

                    if clf != None:
                        models.append({"feature_type" : feature_type, "max_time_lag" : max_time_lag, "percentile" : percentile, "model" : clf, "predictions" : predictions})
                        

        pickle.dump(models, open("one_day_economic_model_{}_{}.pickle.gz".format(pair, feature_type), "wb"))

def train_barrier_model(pair, calendar_df):
    prices, times, volumes = load_time_series(pair, None, False)

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times
    price_df.set_index("times", inplace=True)

    if pair[4:7] == "JPY":
        pip_size = 0.01
    else:
        pip_size = 0.0001

    for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:

        models = []
        for train_offset in [0, 24 * 2]:

            for max_time_lag in [48, 96]:
 
                index = train_offset
                X_train = []
                y_train = []
                while index < len(times):
                    time = times[index]
                    print (index, len(times))

                    curr_calendar_df = calendar_df[(calendar_df["time"] < time) & (calendar_df["time"] > time - (6 * 24 * 60 * 60))]

                    before_prices = (price_df[price_df.index <= time])["prices"].values.tolist()
                    after_prices = (price_df[price_df.index >= time])["prices"].values.tolist()

                    if len(before_prices) < 30 * 24 or len(after_prices) < 30 * 24:
                        index += 24 * 4
                        continue

                    feature_vector = []
                    forecast = []

                    if feature_type == "news_movement_feature_simple":
                        feature_vector = news_movement_feature_simple(time, curr_calendar_df, max_time_lag, before_prices)
                    else:
                        feature_vector = news_movement_feature_ranking(time, curr_calendar_df, max_time_lag, before_prices)

                    barriers = []
                    for is_buy in [True, False]:
                        
                        for barrier in barrier_levels:
                            for ratio in [1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3]:

                                if is_buy:
                                    top_barrier = after_prices[0] + (barrier * ratio * pip_size)
                                    bottom_barrier = after_prices[0] - (barrier * pip_size)
                                else:
                                    top_barrier = after_prices[0] + (barrier * pip_size)
                                    bottom_barrier = after_prices[0] - (barrier * pip_size * ratio)

                                found = None
                                for price in after_prices:
                                    if price >= top_barrier:
                                        found = True
                                        break

                                    if price <= bottom_barrier:
                                        found = False
                                        break

                                if found is not None:
                                    barriers.append(found)
                                else:
                                    barriers.append(after_prices[-1] > after_prices[0])

                    X_train.append(feature_vector)
                    y_train.append(barriers)

                    index += 24 * 4

                y_index = 0
                for is_buy in [True, False]:
                    for barrier in barrier_levels:
                        for ratio in [1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3]:
                            print (is_buy, barrier, ratio)
                            y = [y[y_index] for y in y_train]
                            X = X_train
                            y_index += 1

                            param_list = [
                                None,
                                {"learning_rate" : 0.8794, "n_estimators" : 703, "max_depth" : 3, "gamma" : 0.8492},
                            ]

                            for params in param_list:
                                
                 
                                if params == None:
                                    clf = xgb.XGBClassifier()
                                else:
                                    clf = xgb.XGBClassifier(
                                        max_depth=int(round(params["max_depth"])),
                                        learning_rate=float(params["learning_rate"]),
                                        n_estimators=int(params["n_estimators"]),
                                        gamma=params["gamma"])

                                clf.fit(np.array(X), y)
                                predictions = clf.predict_proba(np.array(X))[:,1]

                            if clf != None:
                                models.append({"max_time_lag" : max_time_lag, "is_buy" : is_buy, "barrier" : barrier, "ratio" : ratio, "model" : clf, "predictions" : predictions})
         
        pickle.dump(models, open("economic_news_model_barrier_model_reward_ratio_{}_{}.pickle.gz".format(pair, feature_type), "wb"))

def train_exceed_model(pair, calendar_df):
    prices, times, volumes = load_time_series(pair, None, False)

    if pair[4:7] == "JPY":
        pip_size = 0.01
    else:
        pip_size = 0.0001

    day_ranges = [0.5, 1.0, 1.5, 2.0, 3, 4, 6, 8, 10, 15, 20, 25, 30] 

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times
    price_df.set_index("times", inplace=True)

    for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:

        models = []
        for train_offset in [0, 24 * 1, 24 * 2, 24 * 3]:

            for max_time_lag in [48, 96]:

                index = train_offset
                X_train = []
                y_train_buy = []
                y_train_sell = []
                while index < len(times):
                    time = times[index]
                    print (index, len(times))

                    curr_calendar_df = calendar_df[(calendar_df["time"] < time) & (calendar_df["time"] > time - (6 * 24 * 60 * 60))]
                    before_prices = (price_df[price_df.index < time])["prices"].values.tolist()
                    after_prices = (price_df[price_df.index >= time])["prices"].values.tolist()

                    if len(before_prices) < 30 * 24 or len(after_prices) < 30 * 4:
                        index += 24 * 4
                        continue


                    forecast_buy = []
                    forecast_sell = []

                    if feature_type == "news_movement_feature_simple":
                        feature_vector = news_movement_feature_simple(time, curr_calendar_df, max_time_lag, before_prices)
                    else:
                        feature_vector = news_movement_feature_ranking(time, curr_calendar_df, max_time_lag, before_prices)

                    for day_range in day_ranges:
                        for barrier in barrier_levels:
                            hour_range = int(day_range * 24)
                            forecast_buy.append((after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0]) > (barrier * pip_size))
                            forecast_sell.append((after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0]) < -(barrier * pip_size))

                    X_train.append(feature_vector)
                    y_train_buy.append(forecast_buy)
                    y_train_sell.append(forecast_sell)

                    index += 24 * 4

                for is_buy in [True, False]:
                    y_index = 0
                    for day_range in day_ranges:
                        for barrier in barrier_levels:
                            print (is_buy, day_range, barrier)

                            if is_buy:
                                y = [y[y_index] for y in y_train_buy]
                            else:
                                y = [y[y_index] for y in y_train_sell]

                            y_index += 1

                            param_list = [
                                None,
                                {"learning_rate" : 0.8794, "n_estimators" : 703, "max_depth" : 3, "gamma" : 0.8492},
                            ]

                            for params in param_list:
                                
                 
                                if params == None:
                                    clf = xgb.XGBClassifier()
                                else:
                                    clf = xgb.XGBClassifier(
                                        max_depth=int(round(params["max_depth"])),
                                        learning_rate=float(params["learning_rate"]),
                                        n_estimators=int(params["n_estimators"]),
                                        gamma=params["gamma"])

                                clf.fit(np.array(X_train), y)
                                predictions = clf.predict_proba(np.array(X_train))[:,1]

                            if clf != None:
                                models.append({"max_time_lag" : max_time_lag, "is_buy" :is_buy, "barrier" : barrier, "day_range" : day_range, "model" : clf, "predictions" : predictions})
                                

        pickle.dump(models, open("economic_news_model_exceed_model_reward_ratio_{}_{}.pickle.gz".format(pair, feature_type), "wb"))


def check_memory():
    import psutil
    import gc
    
    memory = psutil.virtual_memory() 
    while memory.percent > 80:
        gc.collect()
        memory = psutil.virtual_memory() 

def get_curr_calendar_day(day_offset):

    curr_date = datetime.datetime.now(timezone('US/Eastern')).strftime("%b%d.%Y").lower()

    week_day = datetime.datetime.now(timezone('US/Eastern')).weekday()

    print ("curr day", week_day)


    if week_day == 6 or week_day == 0:
        back_day_num = 4
    else:
        back_day_num = 2

    print ("loading...", back_day_num)


    lag_days = 5 + day_offset

    calendar_data = []
    for back_day in range(-1 + day_offset, 1000):
        d = datetime.datetime.now(timezone('US/Eastern')) - datetime.timedelta(days=back_day)

        day_before = d.strftime("%b%d.%Y").lower()
        print (day_before)

        if get_mac() != 150538578859218:
            df = pd.read_csv("/root/news_data/{}.csv".format(day_before))
        else:
            df = pd.read_csv(root_dir + "news_data/{}.csv".format(day_before))

        calendar_data = [df] + calendar_data

        if len(df) > 0:
            min_time = df["time"].min()
            time_lag_compare = calculate_time_diff(time.time(), min_time)
            print ("time lag", time_lag_compare)
            if time_lag_compare >= 24 * lag_days:
                break


    calendar = {"last_check" : time.time(), "day" :  curr_date, "df" : pd.concat(calendar_data)}

    pickle.dump(calendar, open("/tmp/calendar_data_historic_short", 'wb'))

    return calendar["df"]

def store_model_prediction(cnx, pair, model_type, model_group, is_buy, scores):

    cursor = cnx.cursor()

    percentiles = []
    for percentile in range(1, 200):
        percentiles.append(np.percentile(scores, percentile * 0.5))

    query = ("""INSERT INTO signal_summary(timestamp, pair, model_type, model_group, forecast_dir, forecast_percentiles) 
                values (now(),'{}','{}','{}','{}','{}')""".
        format(
            pair,
            model_type,
            model_group,
            is_buy,
            json.dumps(percentiles)
            ))

    cursor.execute(query)
    cnx.commit()

def get_one_day_regression_prediction(pair, pdf, cnx):
    

    check_memory()

    predictions = []
    forecasts = []

    for prev_days in range(5):
        print ("prev days", prev_days)
        curr_calendar_df = get_curr_calendar_day(prev_days)
        curr_time = time.time() - (24 * 60 * 60 * prev_days)
        curr_calendar_df = curr_calendar_df[curr_calendar_df["time"] < curr_time]


        for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:
            if os.path.isfile("{}one_day_economic_model_{}_{}.pickle.gz".format(root_dir, pair, feature_type)) == False:
                continue

            models = pickle.load(open("{}one_day_economic_model_{}_{}.pickle.gz".format(root_dir, pair, feature_type), "rb"))
            print ("here2", pair)

            for max_time_lag in [6, 12, 24, 48, 96]:

                if feature_type == "news_movement_feature_simple":
                    feature_vector = news_movement_feature_simple(curr_time, curr_calendar_df, max_time_lag, None)
                else:
                    feature_vector = news_movement_feature_ranking(curr_time, curr_calendar_df, max_time_lag, None)

                for model in models:
                    if model["max_time_lag"] != max_time_lag:
                        continue

                    forecast = model["model"].predict(np.array([feature_vector]))[0]

                    predictions += list(model["predictions"])
                    forecasts.append(forecast)

    predictions = sorted(predictions)

    scores = []
    for forecast in forecasts:
        forecast_perc = float(bisect(predictions, forecast)) / len(predictions)

        if abs(forecast_perc - 0.5) < 0.06:
            continue

        score = (forecast_perc - 0.5) 
        scores.append(score)

    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    plt.figure(figsize=(5,4))

    is_buy = 0
    if (np.percentile(scores, 55) > 0) == (np.percentile(scores, 45) > 0):
        if np.median(scores) > 0:
            is_buy = 1
            sns.distplot(scores, color="lime", label="Economic Forecast BUY", **kwargs)
        else:
            is_buy = -1
            sns.distplot(scores, color="r", label="Economic Forecast SELL", **kwargs)
    else:
        sns.distplot(scores, color="orange", label="Economic Forecast NEUTRAL", **kwargs)

    store_model_prediction(cnx, pair, "1 Day Forecast", "Economic", is_buy, scores)

    plt.xlim(-1,1)

    plt.title(pair + " One Day Economic Forecast")
    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_1_day_economic_forecast.png".format(pair))
    #pdf.savefig()
    #plt.show()
    plt.close()

def get_regression_prediction(pair, pdf, cnx):

    prices, times = get_time_series(pair, 24 * 20)
    
    check_memory()

    predictions = []
    forecasts = []

    for prev_days in range(5):
        print ("prev days", prev_days)
        curr_calendar_df = get_curr_calendar_day(prev_days)
        curr_time = time.time() - (24 * 60 * 60 * prev_days)
        curr_calendar_df = curr_calendar_df[curr_calendar_df["time"] < curr_time]
        select_prices = prices[24 * prev_days:]

        for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:
            if os.path.isfile("{}regression_economic_model_{}_{}.pickle.gz".format(root_dir, pair, feature_type)) == False:
                continue

            models = pickle.load(open("{}regression_economic_model_{}_{}.pickle.gz".format(root_dir, pair, feature_type), "rb"))
            print ("here1", pair)

            for max_time_lag in [48, 96]:

                if feature_type == "news_movement_feature_simple":
                    feature_vector = news_movement_feature_simple(curr_time, curr_calendar_df, max_time_lag, select_prices)
                else:
                    feature_vector = news_movement_feature_ranking(curr_time, curr_calendar_df, max_time_lag, select_prices)

                for model in models:
                    if model["max_time_lag"] != max_time_lag:
                        continue

                    forecast = model["model"].predict(np.array([feature_vector]))[0]

                    predictions += list(model["predictions"])
                    forecasts.append(forecast)

    if len(predictions) == 0:
        return 

    predictions = sorted(predictions)

    scores = []
    for forecast in forecasts:
        forecast_perc = float(bisect(predictions, forecast)) / len(predictions)

        if abs(forecast_perc - 0.5) < 0.06:
            continue

        score = (forecast_perc - 0.5) 
        scores.append(score)

    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    is_buy = 0
    plt.figure(figsize=(5,4))
    if (np.percentile(scores, 56) > 0) == (np.percentile(scores, 44) > 0):
        if np.median(scores) > 0:
            is_buy = 1
            sns.distplot(scores, color="lime", label="Economic Forecast BUY", **kwargs)
        else:
            is_buy = -1
            sns.distplot(scores, color="r", label="Economic Forecast SELL", **kwargs)
    else:
        sns.distplot(scores, color="orange", label="Economic Forecast NEUTRAL", **kwargs)

    store_model_prediction(cnx, pair, "All Economic News Forecast", "Economic", is_buy, scores)

    plt.xlim(-1,1)

    plt.title(pair + " Regression Economic Forecast")
    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_regression_economic_forecast.png".format(pair))
    #pdf.savefig()
    #plt.show()
    plt.close()

def get_barrier_prediction(pair, pdf, cnx):
    prices, times = get_time_series(pair, 24 * 5)
    
    check_memory()

    curr_calendar_df = get_curr_calendar_day(0)
    curr_time = time.time() 

    predictions = []
    forecasts = []
    probabilities = []

    print ("here3", pair)
    for feature_type in ["news_movement_feature_simple", "news_movement_feature_ranking"]:
        if os.path.isfile("{}economic_news_model_barrier_model_reward_ratio_{}_{}.pickle.gz".format(root_dir, pair, feature_type)) == False:
            continue

        models = pickle.load(open("{}economic_news_model_barrier_model_reward_ratio_{}_{}.pickle.gz".format(root_dir, pair, feature_type), "rb"))

        for max_time_lag in [48, 96]:

            if feature_type == "news_movement_feature_simple":
                feature_vector = news_movement_feature_simple(curr_time, curr_calendar_df, max_time_lag, prices)
            else:
                feature_vector = news_movement_feature_ranking(curr_time, curr_calendar_df, max_time_lag, prices)

            for model in models:
                barrier = model["barrier"]
                ratio = model["ratio"]
                is_buy = model["is_buy"]

                if model["max_time_lag"] != max_time_lag:
                    continue

                prob = model["model"].predict_proba(np.array([feature_vector]))[0][1]

                predictions += list(model["predictions"])
                if is_buy:
                    if prob > 0.5:
                        forecasts.append(barrier * ratio)
                    else:
                        forecasts.append(-barrier)
                else:
                    if prob > 0.5:
                        forecasts.append(barrier)
                    else:
                        forecasts.append(-barrier * ratio)

                probabilities.append(prob)

    if len(predictions) == 0:
        return

    predictions = sorted(predictions)

    scores = []
    for forecast, prob in zip(forecasts, probabilities):
        forecast_perc = float(bisect(predictions, prob)) / len(predictions)

        score = int((forecast_perc - 0.5) * 100)
        scores += [forecast] * abs(score)

    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    store_model_prediction(cnx, pair, "Barrier", "Economic", is_buy, scores)

    plt.figure(figsize=(5,4))
    if np.median(scores) > 0:
        sns.distplot(scores, color="lime", label="Economic Forecast BUY", **kwargs)
    else:
        sns.distplot(scores, color="r", label="Economic Forecast SELL", **kwargs)

    #plt.xlim(-1,1)
    #plt.ylim(0,25)

    plt.title(pair + " Pip Movement Economic Forecast")
    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_barrier_economic_forecast.png".format(pair))
    #pdf.savefig()
    plt.show()
    plt.close()


def create_model():

    df_set = []
    for year in range(2007, 2020):
        print (year, len(get_calendar_df(None, year)))
        df_set.append(get_calendar_df(None, year))

    calendar_df = pd.concat(df_set)

    for pair in currency_pairs:
        train_one_day_regression_model(pair, calendar_df)
        #train_regression_model(pair, calendar_df)
        #train_barrier_model(pair, calendar_df)
        #train_exceed_model(pair, calendar_df)

def get_today_prediction():

    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    with PdfPages('multipage_pdf.pdf') as pdf:
        for pair in currency_pairs:

            get_one_day_regression_prediction(pair, pdf, cnx)
            get_regression_prediction(pair, pdf, cnx)
        

trade_logger = setup_logger('first_logger', root_dir + "predictions_economic_forecast.log")
trade_logger.info('Starting ') 

try:
    get_today_prediction()
    trade_logger.info('Finished ') 
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())

