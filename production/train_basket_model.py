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

checkIfProcessRunning('train_basket_model.py', "")

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

select_lags = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75] + range(2, 15)
barrier_levels = [10, 20, 30, 40, 50, 100, 150, 200, 250]
#select_lags = range(1, 14)

def get_feature_vector(pair, select_lags, price_df, time, trend_map):

    if time is not None:
        before_price_df = price_df[price_df["times"] <= time]
        if len(before_price_df) < 10:
            return []
    else:
        before_price_df = price_df

    price_delta = {}
    count = 0

    select_index = 0
    for i, compare_pair in enumerate(currency_pairs):

        if compare_pair == pair:
            select_index = count

        if pair[0:3] in compare_pair or pair[4:7] in compare_pair:

            before_prices = before_price_df["prices" + str(i)].values.tolist()

            for day_range in select_lags:
                hour_range = int(day_range * 24)
                delta = before_prices[-1] - before_prices[-min(len(before_prices)-1, hour_range)]
                index = bisect(trend_map[compare_pair][day_range], abs(delta))
                price_delta[str(count) + str(day_range)] = (float(index) / len(trend_map[compare_pair][day_range])) 

            count += 1

    feature_vector = []
    for pair_index in range(count):
        if pair_index == select_index:
            continue

        for day_range in select_lags: 

            delta = price_delta[str(select_index) + str(day_range)] - price_delta[str(pair_index) + str(day_range)]
            feature_vector.append(delta)

    return feature_vector

def get_ranked_feature_vector(pair, select_lags, price_df, time, trend_map):

    if time is not None:
        before_price_df = price_df[price_df["times"] <= time]
        if len(before_price_df) < 10:
            return []
    else:
        before_price_df = price_df

    price_delta = {}
    count = 0

    select_index = 0
    for i, compare_pair in enumerate(currency_pairs):

        if compare_pair == pair:
            select_index = count

        if pair[0:3] in compare_pair or pair[4:7] in compare_pair:

            before_prices = before_price_df["prices" + str(i)].values.tolist()

            for day_range in select_lags:
                hour_range = int(day_range * 24)
                delta = before_prices[-1] - before_prices[-min(len(before_prices)-1, hour_range)]
                index = bisect(trend_map[compare_pair][day_range], abs(delta))

                if compare_pair[0:3] == pair[0:3] or compare_pair[4:7] == pair[4:7]:
                    price_delta[str(count) + str(day_range)] = (float(index) / len(trend_map[compare_pair][day_range])) 
                else:
                    price_delta[str(count) + str(day_range)] = -(float(index) / len(trend_map[compare_pair][day_range])) 

            count += 1

    feature_vector = []
    for day_range in select_lags: 

        sorted_z_scores = []
        for pair_index in range(count):

            delta = price_delta[str(pair_index) + str(day_range)]
            sorted_z_scores.append([delta, pair_index])

        sorted_z_scores = sorted(sorted_z_scores, key=lambda x: x[0])

        rank_index = [i for i, v in enumerate(sorted_z_scores) if v[1] == select_index][0]
        feature_vector.append(rank_index)

    return feature_vector

def train_exceed_model(pair):
    prices, times, volumes = load_time_series(pair, None, False)

    if pair[4:7] == "JPY":
        pip_size = 0.01
    else:
        pip_size = 0.0001


    day_ranges = [0.5, 1.0, 1.5, 2.0, 3, 4, 6, 8, 10, 15, 20, 25, 30] 

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times

    for feature_set in ["_mean_revert"]:
        models = []
        for train_offset in [0, 24 * 7]:

            index = train_offset
            X_train = []
            y_train_buy = []
            y_train_sell = []
            while index < len(times):
                time = times[index]
                print (index, len(times))

                before_prices = (price_df[price_df["times"] < time])["prices"].values.tolist()
                after_prices = (price_df[price_df["times"] >= time])["prices"].values.tolist()

                if len(before_prices) < 30 * 24 or len(after_prices) < 30 * 24:
                    index += 24 * 7
                    continue

                feature_vector = []
                forecast_buy = []
                forecast_sell = []

                window_prices = []
                for day_range in select_lags:
                    hour_range = int(day_range * 24)
                    window_prices.append(before_prices[-min(len(before_prices), hour_range)])
                    feature_vector.append(before_prices[-1] - before_prices[-min(len(before_prices), hour_range)])
          
                if feature_set == "_mean_revert":
                    feature_vector = []
                    mean_price = np.mean(window_prices)
                    for day_range in select_lags:
                        hour_range = int(day_range * 24)
                        feature_vector.append(before_prices[-min(len(before_prices), hour_range)] - mean_price)
              
                for day_range in day_ranges:
                    for barrier in barrier_levels:
                        hour_range = int(day_range * 24)
                        forecast_buy.append((after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0]) > (barrier * pip_size))
                        forecast_sell.append((after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0]) < -(barrier * pip_size))

                X_train.append(feature_vector)
                y_train_buy.append(forecast_buy)
                y_train_sell.append(forecast_sell)

                index += 24 * 7

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
                            models.append({"is_buy" :is_buy, "barrier" : barrier, "day_range" : day_range, "model" : clf, "predictions" : predictions})
                            

        pickle.dump(models, open("exceed_model_percentile_{}{}.pickle.gz".format(pair, feature_set), "wb"))

def train_barrier_model(pair, all_price_df, price_delta):
    prices, times, volumes = load_time_series(pair, None, False)

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times

    if pair[4:7] == "JPY":
        pip_size = 0.01
    else:
        pip_size = 0.0001

    for feature_type in ["_ranked"]:
        models = []
        for train_offset in [0, 24 * 7]:
     

            index = train_offset
            X_train = []
            y_train = []
            while index < len(times):
                time = times[index]
                print (index, len(times))

                before_prices = (price_df[price_df["times"] < time])["prices"].values.tolist()
                after_prices = (price_df[price_df["times"] >= time])["prices"].values.tolist()

                if len(before_prices) < 30 * 24 or len(after_prices) < 30 * 24:
                    index += 24 * 7
                    continue

                forecast = []

                if feature_type == "_ranked":
                    feature_vector = get_ranked_feature_vector(pair, select_lags, all_price_df, time, price_delta)
                else:
                    feature_vector = get_feature_vector(pair, select_lags, all_price_df, time, price_delta)

                print (feature_type, len(feature_vector))

                if len(feature_vector) < 10:
                    index += 24 * 7
                    continue

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

                index += 24 * 7

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
                            models.append({"is_buy" : is_buy, "barrier" : barrier, "ratio" : ratio, "model" : clf, "predictions" : predictions})
     
        pickle.dump(models, open("barrier_model_reward_ratio_basket_{}{}.pickle.gz".format(pair, feature_type), "wb"))


def regression_rmse_calculator(X, y, cross_val_num, is_sample_wt, rand_seed_offset, params):

    y_true_indexes = [index for index in range(len(y)) if y[index] > 0]
    y_false_indexes = [index for index in range(len(y)) if y[index] < 0]

    y_test_all = []
    y_preds_all = []
    for iteration in range(cross_val_num):

        # 16 base
        rand.seed(iteration << (rand_seed_offset * 4))
        rand.shuffle(y_true_indexes)

        rand.seed(iteration << (rand_seed_offset * 4))
        rand.shuffle(y_false_indexes)

        min_size = max(15, int(min(len(y_false_indexes), len(y_true_indexes)) * 0.35))
        if min_size >= max(len(y_true_indexes), len(y_false_indexes)) * 0.8:
            return -1

        true_indexes = y_true_indexes[:min_size]
        false_indexes = y_false_indexes[:min_size]

        X_train = []
        y_train = []

        X_test = []
        y_test = []
        for index in range(len(y)):
            if index in true_indexes + false_indexes:
                X_test.append(X[index])
                y_test.append(y[index])
            else:
                X_train.append(X[index])
                y_train.append(y[index])
        
        if params == None:
            clf = xgb.XGBRegressor()
        else:
            clf = xgb.XGBRegressor(
                max_depth=int(round(params["max_depth"])),
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(params["n_estimators"]),
                gamma=params["gamma"])

        if is_sample_wt:

            true_wt = float(sum(y_train)) / len(y_train)
            false_wt = 1 - true_wt

            weights = []
            for y_s in y_train:
                if y_s:
                    weights.append(false_wt)
                else:
                    weights.append(true_wt)

            clf.fit(np.array(X_train), y_train, sample_weight=np.array(weights))
        else:
            clf.fit(np.array(X_train), y_train)

        preds = clf.predict(np.array(X_test))

        y_test_all += list(y_test)
        y_preds_all += list(preds)

    print (y_test_all[:5], y_preds_all[:5])

    return math.sqrt(mean_squared_error(y_test_all, y_preds_all))



def train_regression_model(pair, all_price_df, trend_map):
    prices, times, volumes = load_time_series(pair, None, False)

    price_df = pd.DataFrame()
    price_df["prices"] = prices
    price_df["times"] = times

    for feature_type in ["", "_ranked"]:
        models = []
        for train_offset in [0, 24 * 7]:

            index = train_offset
            X_train = []
            y_train = []
            while index < len(times):
                time = times[index]
                print (index, len(times))

                before_prices = (price_df[price_df["times"] < time])["prices"].values.tolist()
                after_prices = (price_df[price_df["times"] >= time])["prices"].values.tolist()

                if len(before_prices) < 30 * 24 or len(after_prices) < 30 * 24:
                    index += 24 * 7
                    continue

                forecast = []

                if feature_type == "_ranked":
                    feature_vector = get_ranked_feature_vector(pair, select_lags, all_price_df, time, trend_map)
                else:
                    feature_vector = get_feature_vector(pair, select_lags, all_price_df, time, trend_map)

                if len(feature_vector) < 10:
                    index += 24 * 7
                    continue

                for day_range in range(1, 30):
                    hour_range = day_range * 24
                    forecast.append(after_prices[min(len(after_prices)-1, hour_range)] - after_prices[0])

                X_train.append(feature_vector)
                y_train.append(forecast)

                index += 24 * 7

            for day_range in range(29):

                for percentile in [50, 60, 70]:
                    y = [y[day_range] for y in y_train]

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

                        print ("training size", len(X))
                        clf.fit(np.array(X), y)
                        predictions = clf.predict(np.array(X))

                    if clf != None:
                        models.append({"percentile" : percentile, "day_range" : day_range, "model" : clf, "predictions" : predictions})
                        

        pickle.dump(models, open("basket/forecast_model_basket_{}{}.pickle.gz".format(pair, feature_type), "wb"))

def check_memory():
    import psutil
    import gc

    memory = psutil.virtual_memory() 
    while memory.percent > 80:
        gc.collect()
        memory = psutil.virtual_memory() 

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

    print (query)

    cursor.execute(query)
    cnx.commit()

def get_regression_prediction(pair, pdf, all_price_df, cnx, trend_map):

    check_memory()

    predictions = []
    forecasts = []

    for feature_type in ["", "_ranked"]:
        models = pickle.load(open("{}basket/forecast_model_basket_{}{}.pickle.gz".format(root_dir, pair, feature_type), "rb"))

        if feature_type == "_ranked":
            feature_vector = get_ranked_feature_vector(pair, select_lags, all_price_df, time, trend_map)
        else:
            feature_vector = get_feature_vector(pair, select_lags, all_price_df, time, trend_map)

        print ("here1", pair)

        for model in models:

            forecast = model["model"].predict(np.array([feature_vector]))[0]

            predictions += list(model["predictions"])
            forecasts.append(forecast)

    predictions = sorted(predictions)

    scores = []
    for forecast in forecasts:
        forecast_perc = float(bisect(predictions, forecast)) / len(predictions)

        score = (forecast_perc - 0.5) 
        if abs(score) > 0.06:
            scores.append(score)

    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    is_buy = 0
    plt.figure(figsize=(5,4))
    if (np.percentile(scores, 58) > 0) == (np.percentile(scores, 42) > 0):
        if np.median(scores) > 0:
            is_buy = 1
            sns.distplot(scores, color="lime", label="Basket Forecast BUY", **kwargs)
        else:
            is_buy = -1
            sns.distplot(scores, color="r", label="Basket Forecast SELL", **kwargs)
    else:
        sns.distplot(scores, color="orange", label="Basket Forecast NEUTRAL", **kwargs)

    store_model_prediction(cnx, pair, "Forecast", "Basket", is_buy, scores)

    plt.xlim(-1,1)

    plt.title(pair + " Basket Forecast")
    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')

    plt.legend()
    plt.savefig("/var/www/html/images/{}_basket_forecast.png".format(pair))
    #pdf.savefig()
    #plt.show()
    plt.close()

def get_barrier_prediction(pair, pdf, all_price_df, cnx, trend_map):

    check_memory()

    predictions = []
    forecasts = []
    probabilities = []

    for feature_type in ["", "_ranked"]:
        models = pickle.load(open("{}basket/barrier_model_reward_ratio_basket_{}{}.pickle.gz".format(root_dir, pair, feature_type), "rb"))
        print ("here2", pair)

        if feature_type == "_ranked":
            feature_vector = get_ranked_feature_vector(pair, select_lags, all_price_df, time, trend_map)
        else:
            feature_vector = get_feature_vector(pair, select_lags, all_price_df, time, trend_map)

        for model in models:
            barrier = model["barrier"]
            ratio = model["ratio"]
            is_buy = model["is_buy"]

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

    predictions = sorted(predictions)

    scores = []
    for forecast, prob in zip(forecasts, probabilities):
        forecast_perc = float(bisect(predictions, prob)) / len(predictions)

        score = int((forecast_perc - 0.5) * 100)
        scores += [v for v in ([forecast] * abs(score)) if abs(v) > 40]

    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2})

    is_buy = 0
    plt.figure(figsize=(5,4))
    if (np.percentile(scores, 55) > 0) == (np.percentile(scores, 45) > 0):
        if np.median(scores) > 0:
            is_buy = 1
            sns.distplot(scores, color="lime", label="Basket Forecast BUY", **kwargs)
        else:
            is_buy = -1
            sns.distplot(scores, color="r", label="Basket Forecast SELL", **kwargs)
    else:
        sns.distplot(scores, color="orange", label="Basket Forecast NEUTRAL", **kwargs)

    store_model_prediction(cnx, pair, "Barrier", "Basket", is_buy, scores)

    #plt.xlim(-1,1)
    #plt.ylim(0,25)

    plt.title(pair + " Pip Movement Basket")
    plt.xlabel("SELL <-----> BUY")

    plt.axvline(0, color='black')  

    plt.legend()
    plt.savefig("/var/www/html/images/{}_basket_barrier.png".format(pair))
    #pdf.savefig()
    #plt.show()
    plt.close()


def create_model():

    trend_map = {}
    for i, compare_pair in enumerate(currency_pairs):
        prices, times, volumes = load_time_series(compare_pair, None, True)
        price_df2 = pd.DataFrame()
        price_df2["prices" + str(i)] = prices
        price_df2["times"] = times

        if i > 0:
            price_all_df = price_all_df.set_index('times').join(price_df2.set_index('times'), how='inner')
            price_all_df.reset_index(inplace=True)
        else:
            price_all_df = price_df2

        trend_map[compare_pair] = {}
    
        for index in range(1, len(prices), 24):
            for time_index in select_lags:
                hour_range = int(time_index * 24)
                if time_index not in trend_map[compare_pair]:
                    trend_map[compare_pair][time_index] = []

                if index + time_index < len(prices):
                    trend_map[compare_pair][time_index].append(abs((prices[-index] - prices[-min(len(prices)-1, hour_range + 1 + index)])))

        for time_index in select_lags:
            trend_map[compare_pair][time_index] = sorted(trend_map[compare_pair][time_index])

        print ("loaded", compare_pair)

    for pair in currency_pairs:
        train_regression_model(pair, price_all_df, trend_map)
        #train_barrier_model(pair, price_all_df, trend_map)

def get_today_prediction():

    trend_map = {}
    for i, compare_pair in enumerate(currency_pairs):
        prices, times, volumes = load_time_series(compare_pair, None, True)
        price_df2 = pd.DataFrame()
        price_df2["prices" + str(i)] = prices
        price_df2["times"] = times

        if i > 0:
            price_all_df = price_all_df.set_index('times').join(price_df2.set_index('times'), how='inner')
            price_all_df.reset_index(inplace=True)
        else:
            price_all_df = price_df2

        trend_map[compare_pair] = {}
    
        for index in range(1, len(prices), 24):
            for time_index in select_lags:
                hour_range = int(time_index * 24)
                if time_index not in trend_map[compare_pair]:
                    trend_map[compare_pair][time_index] = []

                if index + time_index < len(prices):
                    trend_map[compare_pair][time_index].append(abs((prices[-index] - prices[-min(len(prices)-1, hour_range + 1 + index)])))

        for time_index in select_lags:
            trend_map[compare_pair][time_index] = sorted(trend_map[compare_pair][time_index])

        print ("loaded", compare_pair)

    for i, compare_pair in enumerate(currency_pairs):
        prices, times = get_time_series(compare_pair, 24 * 30)
        before_price_df2 = pd.DataFrame()
        before_price_df2["prices" + str(i)] = prices
        before_price_df2["times"] = times

        if i > 0:
            before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
            before_all_price_df.reset_index(inplace=True)
        else:
            before_all_price_df = before_price_df2

    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    cursor = cnx.cursor()


    query = ("""SELECT pair
        from signal_summary 
        where model_group = 'Basket' 
        group by pair 
        order by min(abs(timediff(timestamp, now()))) desc
        limit 1

        """)

    cursor.execute(query)

    rows = [row for row in cursor]
    pair = rows[0][0]

    with PdfPages('multipage_pdf.pdf') as pdf:

        trade_logger.info(pair) 

        get_regression_prediction(pair, pdf, before_all_price_df, cnx, trend_map)
        get_barrier_prediction(pair, pdf, before_all_price_df, cnx, trend_map)


if get_mac() == 150538578859218:
    create_model()
    sys.exit(0)


trade_logger = setup_logger('first_logger', root_dir + "predictions_basket.log")
trade_logger.info('Starting ') 


try:
    get_today_prediction()
    trade_logger.info('Finished ') 
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())

