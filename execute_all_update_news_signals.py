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
import shap
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
import download_calendar as download_calendar
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json
import bisect

import os


import paramiko
import json

import logging
import os
import enum

class ModelType(enum.Enum): 
    simple_barrier = 1
    complex_barrier = 2
    simple_regression = 3
    complex_regression = 4
    complex_regression_news_impact = 5
    complex_barrier_news_impact = 6
    complex_barrier_ranking = 7
    complex_regression_ranking = 8
    simple_short_forecast = 9


if get_mac() != 150538578859218:
    root_dir = "/root/" 
else:
    root_dir = "/tmp/" 

model_type = sys.argv[2]
if model_type == "barrier":
    model_type = ModelType.simple_barrier
elif model_type == "time_regression":
    model_type = ModelType.complex_regression
elif model_type == "news_momentum":
    model_type = ModelType.simple_regression
elif model_type == "news_impact":
    model_type = ModelType.complex_barrier
elif model_type == "news_reaction_regression":
    model_type = ModelType.complex_regression_news_impact
elif model_type == "news_reaction_barrier":
    model_type = ModelType.complex_barrier_news_impact
elif model_type == "ranking_regression":
    model_type = ModelType.complex_regression_ranking
elif model_type == "ranking_barrier":
    model_type = ModelType.complex_barrier_ranking
elif model_type == "simple_short_forecast":
    model_type = ModelType.simple_short_forecast
else:
    sys.exit(0)

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


trade_logger = setup_logger('first_logger', root_dir + "update_news_release_signals_all" + sys.argv[1].replace(" ", "_") + str(model_type) + ".log")


def get_curr_calendar_day(model_type):

    curr_date = datetime.datetime.now(timezone('US/Eastern')).strftime("%b%d.%Y").lower()

    week_day = datetime.datetime.now(timezone('US/Eastern')).weekday()

    print ("curr day", week_day)

    
    if os.path.isfile("/tmp/calendar_data_historic_short"):
        calendar = pickle.load(open("/tmp/calendar_data_historic_short", 'rb'))

        news_times = calendar["df"]["time"].values.tolist()

        found_recent_news = False
        for news_time in news_times:
            if abs(time.time() - news_time) < 7 * 60 and time.time() > news_time:
                print ("find new news")
                found_recent_news = True


        if abs(time.time() - calendar["last_check"]) < 60 * 60 * 1:
            if len(calendar["df"]) > 0:
                return calendar["df"]


    if week_day == 6 or week_day == 0:
        back_day_num = 4
    else:
        back_day_num = 2

    print ("loading...", back_day_num)

    lag_days = 4
    if model_type == ModelType.simple_regression:
        lag_days = 5

    calendar_data = []
    for back_day in range(-1, back_day_num + 15):
        d = datetime.datetime.now(timezone('US/Eastern')) - datetime.timedelta(days=back_day)

        day_before = d.strftime("%b%d.%Y").lower()
        print (day_before)

        if os.path.isfile(root_dir + "news_data/{}.csv".format(day_before)) == False:
            continue

        if os.path.getsize(root_dir + "news_data/{}.csv".format(day_before)) == 0:
            continue

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



def cross_val_calculator(X, y, cross_val_num, is_sample_wt, rand_seed_offset, params = None):

    y_true_indexes = [index for index in range(len(y)) if y[index] == True]
    y_false_indexes = [index for index in range(len(y)) if y[index] == False]

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
            clf = xgb.XGBClassifier()
        else:
            clf = xgb.XGBClassifier(
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

        preds = clf.predict_proba(np.array(X_test))[:,1]

        y_test_all += y_test
        y_preds_all += list(preds)

    fpr, tpr, thresholds = metrics.roc_curve(y_test_all, y_preds_all)

    return metrics.auc(fpr, tpr), y_test_all, y_preds_all

def regression_rmse_calculator(X, y, cross_val_num, is_sample_wt, rand_seed_offset, params = None):

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

    return math.sqrt(mean_squared_error(y_test_all, y_preds_all)), y_test_all, y_preds_all


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


df_set = []
for year in range(2007, 2020):
    print (year, len(get_calendar_df(None, year)))
    df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)



def bayesian_optimization_classifier(pip_barrier, X_train, y_train, X_last, auc, prob, model_type, is_simple_evaluate):

    if is_simple_evaluate:
        param_list = [
            None
        ]
    elif model_type == ModelType.simple_barrier:
        param_list = [
            None,
            {"learning_rate" : 0.8794, "n_estimators" : 479, "max_depth" : 3, "gamma" : 0.8492},
            {"learning_rate" : 0.87, "n_estimators" : 703, "max_depth" : 3, "gamma" : 1.022},
            {"learning_rate" : 0.01, "n_estimators" : 1000, "max_depth" : 10, "gamma" : 0.0},
            {"learning_rate" : 0.1014, "n_estimators" : 411, "max_depth" : 4, "gamma" : 0.7338},
        ]
    elif model_type == ModelType.complex_barrier_news_impact or model_type == ModelType.complex_barrier_ranking or model_type == ModelType.complex_barrier:
        param_list = [
            None,
            {"learning_rate" : 0.8794, "n_estimators" : 220, "max_depth" : 3, "gamma" : 0.8492},
            {"learning_rate" : 0.01, "n_estimators" : 220, "max_depth" : 10, "gamma" : 0.0},
            {"learning_rate" : 0.1014, "n_estimators" : 220, "max_depth" : 4, "gamma" : 0.7338},
        ]

    if is_simple_evaluate:
        cross_val_num = 6
    else:
        cross_val_num = 10

    prob_num = 0
    prob_denom = 0

    aucs = []
    y_test_all = []
    y_pred_all = []

    barrier_test_all = []
    barrier_actual_all = []
    for param in param_list:
        new_auc, y_test, y_pred = cross_val_calculator(X_train, y_train, cross_val_num, False, 
            len(aucs), params = param)

        if param != None:
            clf = xgb.XGBClassifier(
                max_depth=param["max_depth"],
                learning_rate=param["learning_rate"],
                n_estimators=param["n_estimators"],
                gamma=param["gamma"])
        else:
            clf = xgb.XGBClassifier()


        clf.fit(np.array(X_train), y_train)
        new_prob = clf.predict_proba([X_last])[0][1]
        prob_num += new_prob * new_auc
        prob_denom += new_auc

        y_test_all += y_test
        y_pred_all += y_pred

        barrier_test_all += [pip_barrier if y else -pip_barrier for y in y_test]
        barrier_actual_all += [2 * pip_barrier * (y - 0.5) for y in y_pred]

        aucs.append(new_auc)

    TT = len([1 for actual, pred in zip(y_test_all, y_pred_all) if actual == (pred > 0.5) and actual == True])
    FF = len([1 for actual, pred in zip(y_test_all, y_pred_all) if actual == (pred > 0.5) and actual == False])
    TF = len([1 for actual, pred in zip(y_test_all, y_pred_all) if actual != (pred > 0.5) and pred > 0.5])
    FT = len([1 for actual, pred in zip(y_test_all, y_pred_all) if actual != (pred > 0.5) and pred < 0.5])

    print ("error", TF, FT, TT, FF)
    fpr, tpr, thresholds = metrics.roc_curve(y_test_all, y_pred_all)
    auc = metrics.auc(fpr, tpr)
    rmse = math.sqrt(mean_squared_error(barrier_test_all, barrier_actual_all))
    print ("final", prob_num / prob_denom, auc)

    return prob_num / prob_denom, auc, rmse, TT, FF, TF, FT

def bayesian_optimization_regressor(X_train, y_train, X_last):

    param_list = [
        None,
        {"learning_rate" : 0.8794, "n_estimators" : 703, "max_depth" : 3, "gamma" : 0.8492},
        {"learning_rate" : 0.1014, "n_estimators" : 411, "max_depth" : 4, "gamma" : 0.7338},
    ]

    prob_num = 0
    prob_denom = 0

    rmses = []
    y_test_all = []
    y_pred_all = []
    for param in param_list:
        rmse, y_test, y_pred = regression_rmse_calculator(X_train, y_train, 8, False, 
            len(rmses), params = param)

        if param != None:
            clf = xgb.XGBRegressor(
                max_depth=param["max_depth"],
                learning_rate=param["learning_rate"],
                n_estimators=param["n_estimators"],
                gamma=param["gamma"])
        else:
            clf = xgb.XGBRegressor()


        clf.fit(np.array(X_train), y_train)
        new_prob = clf.predict([X_last])[0]
        prob_num += new_prob * (1.0 / rmse)
        prob_denom += (1.0 / rmse)

        y_test_all += y_test
        y_pred_all += y_pred

        rmses.append(rmse)

    TT = len([1 for actual, pred in zip(y_test_all, y_pred_all) if (actual > 0) == (pred > 0) and (actual > 0)])
    FF = len([1 for actual, pred in zip(y_test_all, y_pred_all) if (actual > 0) == (pred > 0) and (actual < 0)])
    TF = len([1 for actual, pred in zip(y_test_all, y_pred_all) if (actual > 0) != (pred > 0) and pred > 0])
    FT = len([1 for actual, pred in zip(y_test_all, y_pred_all) if (actual > 0) != (pred > 0) and pred < 0])

    print ("error", TF, FT, TT, FF)
    print ("final", prob_num / prob_denom, math.sqrt(mean_squared_error(y_test_all, y_pred_all)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_all, y_pred_all)

    return prob_num / prob_denom, math.sqrt(mean_squared_error(y_test_all, y_pred_all)), r_value ** 2, TT, FF, TF, FT

news_release_stat_df = pd.read_csv("{}news_dist_stats.csv".format(root_dir))
news_release_stat_df.set_index("key", inplace=True)

def news_movement_feature_complex(before_price_df, curr_time, curr_calendar_df, curr_release, price_delta, is_live = False):

    currency_map_std = {}
    currency_map_mean = {}

    before_prices1 = before_price_df.tail(96)
    for price_index, pair in enumerate(currency_pairs):
        price_mean = before_prices1['prices' + str(price_index)].mean()
        price_std = before_prices1['prices' + str(price_index)].std()
        curr_price = before_prices1['prices' + str(price_index)].tail(1).iloc[0]

        v1 = curr_price - price_mean
        v2 = (curr_price - price_mean) / price_std
        
        currency1 = pair[0:3]
        currency2 = pair[4:7]

        if currency1 not in currency_map_std:
            currency_map_std[currency1] = []
            currency_map_mean[currency1] = []

        if currency2 not in currency_map_std:
            currency_map_std[currency2] = []
            currency_map_mean[currency2] = []

        currency_map_std[currency1].append(v2)
        currency_map_mean[currency1].append(v1)

        currency_map_std[currency2].append(-v2)
        currency_map_mean[currency2].append(-v1)

    currency_news1_count = {}
    currency_news2_count = {}
    currency_news3_count = {}
    
    z_score_dists1 = {}
    z_score_dists2 = {}
    for index, row in curr_calendar_df.iterrows():
        time_lag = calculate_time_diff(curr_time, row["time"])

        if time_lag > 24 * 4:
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

    X_last = [price_delta]
    for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:
        X_last.append(np.mean(currency_map_std[currency]))
        X_last.append(np.mean(currency_map_mean[currency]))

        if curr_release["currency"] == currency:
            X_last.append(curr_release["actual"] - curr_release["previous"])
            X_last.append(curr_release["actual"] - curr_release["forecast"])
        else:
            X_last.append(0)
            X_last.append(0)


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

    for price_index, pair in enumerate(currency_pairs):
        prices = before_prices1['prices' + str(price_index)].values.tolist()
        X_last.append(linreg([x for x in range(len(prices))], prices))

    return X_last

def news_movement_feature_ranking(curr_time, curr_calendar_df, curr_release, price_delta, is_live = False):


    X_last = [price_delta]

    z_score_dists1 = {}
    z_score_dists2 = {}


    curr_calendar_df = calendar_df[(calendar_df["time"] < curr_time) & (calendar_df["time"] > curr_time - (60 * 60 * 24 * 6))]

    for index, row in curr_calendar_df.iterrows():

        time_lag = calculate_time_diff(curr_time, row["time"])

        if time_lag > 24 * 4:
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

    return X_last

def news_impact_feature(before_price_df, after_price_df, curr_time, curr_release):

    
    X_last = []
    for before_times in [24, 48, 72, 96, 120]:
        before_prices1 = before_price_df.tail(before_times)

        trends = []
        for price_index, pair in enumerate(currency_pairs):
            if pair[0:3] == curr_release["currency"] or pair[4:7] == curr_release["currency"]:
                prices = before_prices1['prices' + str(price_index)].values.tolist()

                if pair[4:7] == curr_release["currency"]:
                    trends.append(-linreg([x for x in range(len(prices))], prices))
                else:
                    trends.append(linreg([x for x in range(len(prices))], prices))

        X_last.append(np.mean(trends))

    time_diff_hrs = calculate_time_diff(curr_time, curr_release["time"])

    for after_times in range(1, 6):

        time_lapse = max(5, (float(after_times) / 5) * time_diff_hrs)

        after_prices1 = after_price_df.head(int(time_lapse))

        trends = []
        for price_index, pair in enumerate(currency_pairs):
            if pair[0:3] == curr_release["currency"] or pair[4:7] == curr_release["currency"]:
                prices = after_prices1['prices' + str(price_index)].values.tolist()

                if pair[4:7] == curr_release["currency"]:
                    trends.append(-linreg([x for x in range(len(prices))], prices))
                else:
                    trends.append(linreg([x for x in range(len(prices))], prices))

        X_last.append(np.mean(trends))

    return X_last

def news_movement_feature(before_price_df, curr_time, curr_calendar_df, curr_release, price_delta, is_live = False):

    currency_map_std = {}
    currency_map_mean = {}

    before_prices1 = before_price_df.tail(96)

    currency_news1_count = {}
    currency_news2_count = {}
    currency_news3_count = {}
    
    z_score_dists1 = {}
    z_score_dists2 = {}
    for index, row in curr_calendar_df.iterrows():
        time_lag = calculate_time_diff(curr_time, row["time"])

        if time_lag > 24 * 4:
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

    X_last = [price_delta]
    for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:

        if curr_release["currency"] == currency:
            X_last.append(curr_release["actual"] - curr_release["previous"])
            X_last.append(curr_release["actual"] - curr_release["forecast"])
        else:
            X_last.append(0)
            X_last.append(0)


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

    return X_last

def news_movement_feature_simple(curr_time, curr_calendar_df, curr_release, price_delta, is_live = False):


    currency_news1_count = {}
    currency_news2_count = {}
    currency_news3_count = {}
    
    z_score_dists1 = {}
    z_score_dists2 = {}
    for index, row in curr_calendar_df.iterrows():
        time_lag = calculate_time_diff(curr_time, row["time"])

        if time_lag > 24 * 4:
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

    X_last = [price_delta]
    for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:

        if curr_release["currency"] == currency:
            X_last.append(curr_release["actual"] - curr_release["previous"])
            X_last.append(curr_release["actual"] - curr_release["forecast"])
        else:
            X_last.append(0)
            X_last.append(0)


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

    return X_last


def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det

def linreg1(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det

def classification_model_sharpe(barrier_index, X_train_map, y_train_map, forecast):


    X = X_train_map[barrier_index]
    y = y_train_map[barrier_index]

    clf = xgb.XGBClassifier()
    clf.fit(np.array(X), y)
    predictions = clf.predict_proba(np.array(X))[:,1]
    lower_percentile = np.percentile([abs(p - 0.5) for p in predictions], 70)
    upper_percentile = np.percentile([abs(p - 0.5) for p in predictions], 90)

    threshold = min(lower_percentile, abs(forecast - 0.5) )

    start = 0
    cross_fold_num = 5
    end = len(X_train_map[barrier_index]) / cross_fold_num

    returns = []
    for i in range(cross_fold_num):
        clf = xgb.XGBClassifier()
        clf.fit(np.array(X[:start] + X[end:]), y[:start] + y[end:])
        trend_preds = clf.predict_proba(X[start:end])[:,1]


        for predict, actual in zip(trend_preds, y[start:end]):
            actual_barrier = (5 + (5 * barrier_index))
    
            if abs(predict - 0.5) > threshold:
                if (predict > 0.5) == actual:
                    returns.append((abs(actual_barrier) - 5) * abs(predict - 0.5)) 
                else:
                    returns.append((-abs(actual_barrier) - 5) * abs(predict - 0.5)) 


        start += len(X_train_map[barrier_index]) / cross_fold_num
        end += len(X_train_map[barrier_index]) / cross_fold_num

    if len(returns) > 20:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 1

    return sharpe

global_ratios = []

def predict_short_forecast_regression_model(X_train_map, y_train_map, X_last, currency_pair, stat_dict, currency, price_df, pip_size, description):

    for barrier_index in y_train_map:

        if barrier_index not in y_train_map:
            continue

        if len(y_train_map[barrier_index]) < 30:
            continue

        X = X_train_map[barrier_index]
        y = y_train_map[barrier_index]

        clf = xgb.XGBRegressor()
        clf.fit(np.array(X), y)

        forecast = clf.predict([X_last])[0]

        time_index = 0
        global_trends = []
        hours_after = barrier_index
        future_prices = price_df['prices'].values.tolist()
        while time_index + hours_after < len(future_prices):
            global_trends.append(abs(future_prices[time_index + hours_after] - future_prices[time_index]) / pip_size)
            time_index += hours_after

        global_trends = sorted(global_trends)

        stat_dict[currency_pair].append({
            "forecast" : forecast,
            "time_wait" : barrier_index,
            "currency" : currency,
            "global_percentile" : float(bisect.bisect(global_trends, abs(forecast))) / len(global_trends),
            "description" : description,
            })

def predict_regression_model(X_train_map, y_train_map, X_last, currency_pair, stat_dict, currency, price_df, pip_size, description):

    for barrier_index in y_train_map:

        if barrier_index not in y_train_map:
            continue

        if len(y_train_map[barrier_index]) < 30:
            continue

        X = X_train_map[barrier_index]
        y = y_train_map[barrier_index]

        clf = xgb.XGBRegressor()
        clf.fit(np.array(X), y)
        predictions = clf.predict(np.array(X))
    
        percentile_thresholds = {}
        for percentile in [70, 75, 80, 85, 90, 95]:
            threshold = np.percentile([(abs(p)) for p in predictions], percentile)
            percentile_thresholds[percentile] = threshold

        lower_percentile = percentile_thresholds[70]
        upper_percentile = percentile_thresholds[85]

        forecast = clf.predict([X_last])[0]

        threshold = max(lower_percentile, min(upper_percentile, abs(forecast)))
        if (abs(forecast)) < lower_percentile:

            stat_dict[currency_pair].append({
                "forecast" : forecast,
                "rmse" : -1,
                "time_wait" : barrier_index,
                "sharpe" : -1,
                "r_2" : -1,
                "currency" : currency,
                "description" : description,
                })

            continue


        #rmse, y_test, y_pred = regression_rmse_calculator(X, y, 8, False, 0)
        forecast, rmse, r_2, TT, FF, TF, FT = bayesian_optimization_regressor(X, y, X_last)

        time_index = 0
        global_trends = []
        hours_after = barrier_index
        future_prices = price_df['prices'].values.tolist()
        while time_index + hours_after < len(future_prices):
            global_trends.append((future_prices[time_index + hours_after] - future_prices[time_index]) / pip_size)
            time_index += hours_after

        print ("r_2", r_2)
        global_ratios.append(r_2)
        print ("mean_ratio", np.mean(r_2))

        #trade_logger.info('Ratio ' + str(abs(forecast) / rmse)) 

        stat_dict[currency_pair].append({
            "forecast" : forecast,
            "rmse" : rmse,
            "time_wait" : barrier_index,
            "sharpe" : -1,
            "r_2" : r_2,
            "currency" : currency,
            "std_globals" : np.std(global_trends),
            "std_actuals" : np.std(y),
            "description" : description,
            "TT" : TT,
            "FF" : FF,
            "FT" : FT,
            "TF" : TF,
            "70th_percentile" : percentile_thresholds[70],
            "75th_percentile" : percentile_thresholds[75],
            "80th_percentile" : percentile_thresholds[80],
            "85th_percentile" : percentile_thresholds[85],
            "90th_percentile" : percentile_thresholds[90],
            "95th_percentile" : percentile_thresholds[95],
            })

def remove_correlated(X_train_map, y_train_map):

    print ("in")
    select_features = []
    for barrier_index in X_train_map:
        X = pd.DataFrame(X_train_map[barrier_index])

        if len(select_features) == 0:
            threshold = 0.9
            col_corr = set() # Set of all the names of deleted columns
            corr_matrix = X.corr()

            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i] # getting the name of column
                        col_corr.add(colname)
                        if colname in X.columns:
                            del X[colname] # deleting the column from the dataset

            select_features = list([column for column in X.columns if column not in col_corr])
            X_train_map[barrier_index] = X.values.tolist()

        else:
            X_train_map[barrier_index] = X[select_features].values.tolist()

    print ("out")

    return X_train_map, select_features


def train_barrier_model(barrier_indexes, pip_size, curr_barrier_interval, price_df, hour_lag, test_calendar, barrier_models, barrier_model_scores, X_train_map, y_train_map, target_barrier = None):

    prev_release_time = 0
    for index2, calendar_row in test_calendar.iterrows():

        if abs(calendar_row['time'] - prev_release_time) < 24 * 60 * 60 * 5:
            continue

        prev_release_time = calendar_row['time']
        future_price_df = price_df[price_df['times'] >= calendar_row['time']]
        if len(future_price_df) == 0:
            continue

        future_prices = (future_price_df)['prices'].values.tolist()
        future_times = (future_price_df)['times'].head(hour_lag + 100).values.tolist()
        if hour_lag >= len(future_times) or abs(future_times[0] - calendar_row['time']) > 60 * 60 * 96:
            continue

        time_lag_compare = calculate_time_diff(future_times[hour_lag], calendar_row['time'])
        if time_lag_compare > 48 + hour_lag:
            continue

        diff_lag_time = time_lag_compare - hour_lag
        new_hour_lag = min(len(future_times) - 1, int(hour_lag - diff_lag_time))
        if new_hour_lag < 0:
            continue

        if new_hour_lag != hour_lag:
            time_lag_compare = calculate_time_diff(future_times[new_hour_lag], calendar_row['time']) - hour_lag
        
            if abs(time_lag_compare) > 1:
                for new_hour_lag in range(len(future_times)):
                    time_lag_compare = hour_lag - calculate_time_diff(future_times[new_hour_lag], calendar_row['time'])
                    if time_lag_compare <= 1:
                        break 
                        
                if time_lag_compare < 0:
                    new_hour_lag = new_hour_lag - 1

                if abs(time_lag_compare) > 30:
                    continue

        if new_hour_lag >= len(future_prices):
            continue

        feature1 = (calendar_row['actual'] - calendar_row['forecast'])
        feature2 = (calendar_row['actual'] - calendar_row['previous'])

        if model_type == ModelType.simple_short_forecast or model_type == ModelType.simple_barrier or model_type == ModelType.simple_regression:
            X_last = [feature1, feature2, future_prices[new_hour_lag] - future_prices[0], calendar_row['actual'], calendar_row['forecast'], calendar_row['previous']]

        elif model_type == ModelType.complex_regression or model_type == ModelType.complex_barrier or \
            model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_barrier_news_impact or \
            model_type == ModelType.complex_regression_ranking or model_type == ModelType.complex_barrier_ranking:

            after_news_time = future_times[new_hour_lag]
            before_price_df = price_df[price_df['times'] <= after_news_time]
            curr_calendar_df = calendar_df[(calendar_df["time"] < after_news_time) & (calendar_df["time"] > after_news_time - (6 * 24 * 60 * 60))]

            if model_type == ModelType.complex_regression_ranking or model_type == ModelType.complex_barrier_ranking:
                X_last = news_movement_feature_ranking(after_news_time, curr_calendar_df, calendar_row, future_prices[new_hour_lag] - future_prices[0])
            elif model_type == ModelType.complex_regression:
                X_last = news_movement_feature(before_price_df, after_news_time, curr_calendar_df, calendar_row, future_prices[new_hour_lag] - future_prices[0])
            elif model_type == ModelType.complex_barrier:
                X_last = news_movement_feature_simple(after_news_time, curr_calendar_df, calendar_row, future_prices[new_hour_lag] - future_prices[0])
            else:
                before_price_df = price_df[price_df['times'] < calendar_row["time"]]
                after_price_df = price_df[price_df['times'] >= calendar_row["time"]]
                X_last = news_impact_feature(before_price_df, after_price_df, after_news_time, calendar_row)

        start_price = future_prices[new_hour_lag]


        for barrier_index in barrier_indexes:

            if barrier_index in barrier_models:
                continue

            if curr_barrier_interval != None and (barrier_index % 2) != curr_barrier_interval:
                continue

            end_price = future_prices[min(len(future_prices) - 1, new_hour_lag + barrier_index)]

            if barrier_index not in y_train_map:
                y_train_map[barrier_index] = []
                X_train_map[barrier_index] = []

            if model_type == ModelType.simple_barrier or model_type == ModelType.complex_barrier or model_type == ModelType.complex_barrier_news_impact or model_type == ModelType.complex_barrier_ranking:
                barrier_pip_size = pip_size * 5
                top_barrier = start_price + (barrier_pip_size + (barrier_pip_size * barrier_index))
                bottom_barrier = start_price - (barrier_pip_size + (barrier_pip_size * barrier_index))

                max_delta = 0
                max_dir = None
                found_barrier = False
                end_time = min(len(future_prices), new_hour_lag + (24 * 20))
                for price in future_prices[new_hour_lag:end_time]:

                    if abs(price - start_price) > max_delta:
                        max_delta = abs(price - start_price)
                        max_dir = price > start_price
                    
                    if price >= top_barrier:
                        y_train_map[barrier_index].append(True)
                        X_train_map[barrier_index].append(X_last)
                        found_barrier = True
                        break

                    if price <= bottom_barrier:
                        y_train_map[barrier_index].append(False)
                        X_train_map[barrier_index].append(X_last)
                        found_barrier = True
                        break

                if found_barrier == False:
                    y_train_map[barrier_index].append(max_dir)
                    X_train_map[barrier_index].append(X_last)
                    # We reached the maximum range within a range 
                    # barriers are sorted so no need to keep going
                    break

            elif model_type == ModelType.simple_short_forecast or model_type == ModelType.complex_regression or model_type == ModelType.simple_regression or model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_regression_ranking:
                if (new_hour_lag + barrier_index) >= len(future_prices):
                    continue

                y_train_map[barrier_index].append((end_price - start_price) / pip_size)
                X_train_map[barrier_index].append(X_last)
            elif model_type == ModelType.time_classification:
                if (new_hour_lag + barrier_index) >= len(future_prices):
                    continue

                y_train_map[barrier_index].append(end_price > start_price)
                X_train_map[barrier_index].append(X_last)

    #X_train_map, select_features = remove_correlated(X_train_map, y_train_map)

    if model_type == ModelType.simple_barrier or model_type == ModelType.complex_barrier or model_type == ModelType.complex_barrier_news_impact or model_type == ModelType.complex_barrier_ranking:
        for barrier_index in y_train_map:

            if barrier_index in barrier_models:
                continue

            if barrier_index not in y_train_map:
                continue

            if len(y_train_map[barrier_index]) < 40:
                continue

            barrier_model_scores[barrier_index] = 0.51
            barrier_clf = xgb.XGBClassifier(seed=1)
            barrier_clf.fit(np.array(X_train_map[barrier_index]), y_train_map[barrier_index])
            barrier_models[barrier_index] = barrier_clf

    return barrier_models, barrier_model_scores, X_train_map, y_train_map

global_aucs = []

def evaluate_barrier_models(X_train_map, y_train_map, row, currency_pair, X_last, barrier_index, model_type, stat_dict, barrier_models, barrier_model_scores):


    prob = barrier_models[barrier_index].predict_proba([X_last])[0][1]
    pip_barrier = (5 + (5 * barrier_index))

    if abs(prob - 0.5) < 0.25:

        stat_dict[currency_pair].append({
                "currency_pair" : currency_pair,
                "currency" : row["currency"],
                "release_time" : row["time"],
                "forecast" : pip_barrier * (prob - 0.5) * 2,
                "barrier" : pip_barrier,
                "probability" : prob,
                "auc" : -1, 
                "rmse" : -1, 
                "description" : row["description"],
                "sharpe" : 1,
                })

        return stat_dict

    prob, auc, rmse, TT, FF, TF, FT = bayesian_optimization_classifier(pip_barrier, X_train_map[barrier_index], y_train_map[barrier_index], X_last, barrier_model_scores[barrier_index], prob, model_type, False)

    #sharpe = classification_model_sharpe(barrier_index, X_train_map, y_train_map, prob)
    sharpe = 1

    print ("Barrier", barrier_index, prob, auc, rmse, pip_barrier * (prob - 0.5) * 2)

    global_aucs.append(auc)

    if sharpe > 0 and auc > 0.45:
        #trade_logger.info('Ratio ' + str(abs(prob - 0.5) / (1.1 - barrier_model_scores[barrier_index]))) 

        stat_dict[currency_pair].append({
                "currency_pair" : currency_pair,
                "currency" : row["currency"],
                "release_time" : row["time"],
                "forecast" : pip_barrier * (prob - 0.5) * 2,
                "barrier" : pip_barrier,
                "probability" : prob,
                "auc" : auc, 
                "rmse" : rmse, 
                "description" : row["description"],
                "sharpe" : sharpe,
                "type" : "M1",
                "TT" : TT,
                "FF" : FF,
                "TF" : TF,
                "FT" : FT
                })

    return stat_dict

def check_monthly_trend(month_trend, test_calendar):

    trends = []
    for index2, calendar_row in test_calendar.iterrows():
        before_price_df = price_df[price_df['times'] <= calendar_row['time']].tail(24 * 20)
        a, b = linreg1(range(len(before_prices)),before_prices)
        trends.append(abs(a))

    upper_percentile = np.percentile(trends, 50)

    return abs(month_trend) > upper_percentile

def get_price_trends(currency_pair):

    before_prices, times = get_time_series(currency_pair, 5 * 4 * 5, "D")

    if currency_pair[4:7] == "JPY":
        pip_size = 0.01
    else:
        pip_size = 0.0001

    trend_map = {}
    for time_index in [1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80]:
        trend_map[time_index] = (before_prices[-1] - before_prices[-time_index-1]) / pip_size

    return trend_map

def get_chosen_barriers(currency_pair):

    if get_mac() == 150538578859218:
        file_dir = "/Users/andrewstevens/Downloads/economic_calendar/metadata/chosen_barriers_{}.pickle".format(currency_pair)
    else:
        file_dir = "/root/trading_data/chosen_barriers_{}.pickle".format(currency_pair)

    if os.path.isfile(file_dir):
        chosen_barriers_map = pickle.load(open(file_dir, "rb"))
    else:
        chosen_barriers_map = None

    for hour in chosen_barriers_map:
        chosen_barriers_map[hour] = sorted(list(set([max(1, v) for v in chosen_barriers_map[hour]])))

    return chosen_barriers_map

def get_chosen_time_horizons(currency_pair):

    if get_mac() == 150538578859218:
        file_dir = "/Users/andrewstevens/Downloads/economic_calendar/metadata/chosen_time_horizons_{}.pickle".format(currency_pair)
    else:
        file_dir = "/root/trading_data/chosen_time_horizons_{}.pickle".format(currency_pair)

    if os.path.isfile(file_dir):
        chosen_barriers_map = pickle.load(open(file_dir, "rb"))
    else:
        chosen_barriers_map = None

    for hour in chosen_barriers_map:
        chosen_barriers_map[hour] = list(set([max(24, v) for v in chosen_barriers_map[hour]]))

    return chosen_barriers_map

def back_test_news_calendar(select_pairs, model_type, only_relevant_currency = False):

    news_summary = []
    stat_dict = {}
    diff = 0

    if model_type == ModelType.simple_barrier:
        prefix_dir = "news_signal_"
    elif model_type == ModelType.simple_regression:
        prefix_dir = "news_momentum_"
    elif model_type == ModelType.complex_regression:
        prefix_dir = "time_regression_"
    elif model_type == ModelType.complex_barrier:
        prefix_dir = "news_impact_"
    elif model_type == ModelType.complex_regression_news_impact:
        prefix_dir = "news_reaction_regression_"
    elif model_type == ModelType.complex_barrier_news_impact:
        prefix_dir = "news_reaction_barrier_"
    elif model_type == ModelType.complex_regression_ranking:
        prefix_dir = "ranking_regression_"
    elif model_type == ModelType.complex_barrier_ranking:
        prefix_dir = "ranking_barrier_"
    elif model_type == ModelType.simple_short_forecast:
        prefix_dir = "simple_short_forecast_"

    for currency_pair in select_pairs:

        curr_calendar_df = get_curr_calendar_day(model_type)

        before_prices, _ = get_time_series(currency_pair, 24 * 20)
        a, b = linreg1(range(len(before_prices)),before_prices)
        month_std = np.std([before_prices[index] - ((a * index) + b) for index in range(len(before_prices))])
        month_trend = abs(a)

        if model_type == ModelType.simple_short_forecast:
            barrier_indexes = [v for v in range(6, 54, 6)]
            curr_barrier_interval = None
            chosen_barrier_map = None
        elif model_type == ModelType.complex_regression or model_type == ModelType.simple_regression or model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_regression_ranking:
            chosen_barrier_map = get_chosen_time_horizons(currency_pair)
            if chosen_barrier_map == None:
                barrier_indexes = [v * 24 for v in range(2, 26)]
            else:
                curr_barrier_interval = None 
        else:
            chosen_barrier_map = get_chosen_barriers(currency_pair)
            if chosen_barrier_map == None:
                barrier_indexes = [v for v in range(1, 21)]
            else:
                curr_barrier_interval = None 


        trade_logger.info('Starting ' + currency_pair + ', Barrier Interval' + str(curr_barrier_interval)) 
        print (currency_pair, "curr barrier interval", curr_barrier_interval)

        curr_time = time.time()
        if model_type == ModelType.complex_regression or model_type == ModelType.complex_barrier_news_impact or \
            model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_barrier_news_impact:

            for i, compare_pair in enumerate(currency_pairs):
                prices, times, volumes = load_time_series(compare_pair, None, True)
                price_df2 = pd.DataFrame()
                price_df2["prices" + str(i)] = prices
                price_df2["times"] = times

                prices, times = get_time_series(compare_pair, 96)
                before_price_df2 = pd.DataFrame()
                before_price_df2["prices" + str(i)] = prices
                before_price_df2["times"] = times

                if i > 0:
                    price_all_df = price_all_df.set_index('times').join(price_df2.set_index('times'), how='inner')
                    price_all_df.reset_index(inplace=True)

                    before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
                    before_all_price_df.reset_index(inplace=True)
                else:
                    price_all_df = price_df2
                    before_all_price_df = before_price_df2

        prices, times, volumes = load_time_series(currency_pair, None, True)
        price_df = pd.DataFrame()
        price_df["prices"] = prices
        price_df["times"] = times
        price_df["volume_buy"] = volumes
        price_df["volume_sell"] = volumes
        price_df.reset_index(inplace=True)

        if model_type == ModelType.complex_regression or model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_barrier_news_impact:
            price_df = price_df.set_index('times').join(price_all_df.set_index('times'), how='inner')
            price_df.reset_index(inplace=True)

        if currency_pair not in stat_dict:
            stat_dict[currency_pair] = []

        for index, row in curr_calendar_df.iterrows():

            print ("global auc", np.mean(global_aucs))

            is_relevant_currency = (currency_pair[0:3] == row["currency"]) or (currency_pair[4:7] == row["currency"])
            if only_relevant_currency and is_relevant_currency == False:
                continue
       
            if model_type == ModelType.simple_barrier or model_type == ModelType.simple_regression or model_type == ModelType.simple_short_forecast:
                curr_time = time.time()

            time_lag = calculate_time_diff(curr_time, row["time"])
            hour_lag = int(round(time_lag))

            if (hour_lag > 24) and model_type == ModelType.simple_short_forecast:
                print ("hour lag > 24", hour_lag, row['description'], curr_time, row["time"])
                continue

            if (hour_lag < 6 or hour_lag > 96) and model_type == ModelType.simple_barrier:
                print ("hour lag > 24", hour_lag, row['description'], curr_time, row["time"])
                continue

            if (hour_lag < 12 or hour_lag > 96) and model_type == ModelType.simple_regression:
                print ("hour lag > 24", hour_lag, row['description'], curr_time, row["time"])
                continue

            if (hour_lag < 12 or hour_lag > 48 + 24) and model_type == ModelType.complex_regression:
                print ("hour lag > 24", hour_lag, row['description'], curr_time, row["time"])
                continue

            if (hour_lag < 12 or hour_lag > 48 + 32) and model_type == ModelType.complex_barrier:
                print ("hour lag > 24", hour_lag, row['description'], curr_time, row["time"])
                continue

            if (hour_lag < 48 or hour_lag > 48 + 32) and (model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_barrier_news_impact or model_type == ModelType.complex_regression_ranking or model_type == ModelType.complex_barrier_ranking):
                print ("hour lag > 24", hour_lag, row['description'], curr_time, row["time"])
                continue

            print ("currency", row["currency"], hour_lag, row['description'])

            if model_type == ModelType.simple_barrier or model_type == ModelType.simple_regression \
                or model_type == ModelType.complex_barrier_news_impact or model_type == ModelType.complex_regression_news_impact  \
                or model_type == ModelType.complex_barrier_ranking or model_type == ModelType.complex_regression_ranking \
                or model_type == ModelType.simple_short_forecast:

                test_calendars = [calendar_df[(calendar_df["description"] == row["description"]) & (calendar_df["currency"] == row["currency"])]]
            elif model_type == ModelType.complex_regression or model_type == ModelType.complex_barrier:
                test_calendar1 = calendar_df[(calendar_df["description"] == row["description"])]
                test_calendar3 = test_calendar1[(test_calendar1["impact"] == row["impact"])]
                test_calendars = [test_calendar3]

            for test_calendar in test_calendars:
                
                if len(test_calendar) < 50:
                    print ("too few samples ", len(test_calendar))
                    continue

                print (currency_pair, row["currency"])

                if currency_pair[4:7] == "JPY":
                    pip_size = 0.01
                else:
                    pip_size = 0.0001

                start_time = time.time()

                if chosen_barrier_map != None:
                    barrier_indexes = chosen_barrier_map[int(hour_lag)]

                #trade_logger.info('chosen barriers ' + str(barrier_indexes)) 

                barrier_models, barrier_model_scores, X_train_map, y_train_map = train_barrier_model(barrier_indexes, pip_size, curr_barrier_interval, price_df, hour_lag, test_calendar, {}, {}, {}, {})

                if len(X_train_map) == 0:
                    continue

                minute_lag = int(round(time_lag * 60))

                if minute_lag < 5000:
                    prices, times = get_time_series(currency_pair, minute_lag, granularity="M1")
                    time_index = minute_lag
                else:
                    prices, times = get_time_series(currency_pair, hour_lag + 48)
                    time_index = hour_lag
        
                print ("curr_time", curr_time)
                print ("start time", (row["time"] - times[-time_index]) / (60 * 60))
                print ("FIRST time", (curr_time - times[-1]) / (60 * 60))
                time_diff = (row["time"] - times[-time_index]) 

                if abs(time_diff) > 70:

                    if minute_lag + (60 * 48) < 5000:
                        prices, times = get_time_series(currency_pair, minute_lag + (60 * 48), granularity="M1")
                        time_index = minute_lag
                    else:
                        prices, times = get_time_series(currency_pair, hour_lag + 48)
                        time_index = hour_lag

                    if row["time"] > times[-time_index]:
                        while time_index > 0:
                            if row["time"] <= times[-time_index]:
                                break
                            time_index -= 1
                    else:
                        while time_index < len(times):
                            if row["time"] >= times[-time_index]:
                                break
                            time_index += 1

                print ("found time", abs(row["time"] - times[-time_index]) / (60 * 60))
                if abs(row["time"] - times[-time_index]) / (60 * 60) > 1:
                    trade_logger.info('Mismatched Time Offset ' + str(abs(row["time"] - times[-time_index]) / (60 * 60))) 
                    continue

                feature1 = (row['actual'] - row['forecast'])
                feature2 = (row['actual'] - row['previous'])

                if model_type == ModelType.simple_barrier or model_type == ModelType.simple_regression or model_type == ModelType.simple_short_forecast:
                    X_last = [feature1, feature2, prices[-1] - prices[-time_index], row['actual'], row['forecast'], row['previous']]
                else:

                    if model_type == ModelType.complex_barrier_ranking or model_type == ModelType.complex_regression_ranking:
                        X_last = news_movement_feature_ranking(curr_time, curr_calendar_df, row, prices[-1] - prices[-time_index], is_live = True)
                    elif model_type == ModelType.complex_regression:
                        X_last = news_movement_feature(before_all_price_df, curr_time, curr_calendar_df, row, prices[-1] - prices[-time_index], is_live = True)
                    elif model_type == ModelType.complex_barrier:
                        X_last = news_movement_feature_simple(curr_time, curr_calendar_df, row, prices[-1] - prices[-time_index], is_live = True)
                    else:
                        b_price_df = before_all_price_df[before_all_price_df["times"] < row["time"]]
                        a_price_df = before_all_price_df[before_all_price_df["times"] >= row["time"]]
                        X_last = news_impact_feature(b_price_df, a_price_df, curr_time, row)

                if model_type == ModelType.simple_short_forecast:
                    predict_short_forecast_regression_model(X_train_map, y_train_map, X_last, currency_pair, stat_dict, row['currency'], price_df, pip_size, row["description"])
                elif model_type == ModelType.complex_regression or model_type == ModelType.simple_regression or model_type == ModelType.complex_regression_news_impact or model_type == ModelType.complex_regression_ranking:
                    predict_regression_model(X_train_map, y_train_map, X_last, currency_pair, stat_dict, row['currency'], price_df, pip_size, row["description"])
                else:
                    for barrier_index in barrier_models:
                        stat_dict = evaluate_barrier_models(X_train_map, y_train_map, row, currency_pair, X_last, barrier_index, model_type, stat_dict, barrier_models, barrier_model_scores)


        if currency_pair in stat_dict:
            if only_relevant_currency == False:
                pickle.dump({"price_trends" : get_price_trends(currency_pair), "month_trend" : month_trend, "month_std" : month_std, "last_barrier_index" : curr_barrier_interval, currency_pair : stat_dict[currency_pair]}, open(root_dir + prefix_dir + "all_" + currency_pair + ".pickle", "wb"))

            pickle.dump({"month_trend" : month_trend, "month_std" : month_std, currency_pair : [item for item in stat_dict[currency_pair] if ((item["currency"] == currency_pair[0:3]) or (item["currency"] == currency_pair[4:7])) ]}, open(root_dir + prefix_dir + currency_pair + ".pickle", "wb"))


    return stat_dict


import psutil

def checkIfProcessRunning(processName, command1, command2):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 4 and processName.lower() in cmdline[2] and command1 == cmdline[3] and command2 == cmdline[4]:
                count += 1
            elif len(cmdline) > 3 and processName.lower() in cmdline[1] and command1 == cmdline[2] and command2 == cmdline[3]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 2:
        sys.exit(0)


checkIfProcessRunning('execute_all_update_news_signals.py', sys.argv[1], sys.argv[2])


def process_demo_pairs():


    pairs = sys.argv[1].split(",")
    stat_dict = back_test_news_calendar(pairs, model_type)

    trade_logger.info('Finished ' + str(stat_dict.keys())) 

def process_managed_account_pairs():


    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    cursor = cnx.cursor()
    query = ("SELECT t1.*, t2.is_hedged FROM managed_strategies t1, managed_accounts t2 where t1.user_id=61 and t1.user_id=t2.user_id and t1.account_nbr=t2.account_nbr order by strategy desc")

    cursor.execute(query)

    setup_rows = []
    for row1 in cursor:
        setup_rows.append(row1)

    cursor.close()

    processed_pairs = set()
    for row in setup_rows:

        user_id = row[0]
        account_nbr = row[1]
        api_key = row[2]
        select_pair = row[3]
        strategy = row[4]
        is_demo = row[5]
        is_max_barrier = row[6]
        strategy_weight = row[7]
        is_hedge = row[8]

        if strategy != "S1" and strategy != "S2":
            continue

        if select_pair in processed_pairs:
            continue

        print (select_pair)

        stat_dict = back_test_news_calendar([select_pair], model_type, only_relevant_currency = (strategy == "S1"))
        processed_pairs.add(select_pair)

    trade_logger.info('Finished ') 

def process_portfolio_account():

    for select_pair in ['GBP_JPY', 'EUR_CHF', 'NZD_CAD', 'USD_JPY', 'GBP_CHF', 'USD_CHF', 'AUD_CHF', 'GBP_USD', 'GBP_NZD', 'EUR_AUD', 'EUR_GBP', 'AUD_CAD', 'NZD_USD']:

        stat_dict = back_test_news_calendar([select_pair], model_type, only_relevant_currency = True)

    trade_logger.info('Finished ') 

try:
    if sys.argv[1] == "account_signals":
        process_managed_account_pairs()
    elif sys.argv[1] == "portfolio_signals":
        process_portfolio_account()
    else:
        process_demo_pairs()
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())

print ("global auc", np.mean(global_aucs))

