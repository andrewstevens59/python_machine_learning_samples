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
import scipy.optimize as sco
import numpy as np
import shap
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
#matplotlib.use('Agg')

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


def get_curr_calendar_day():

    curr_date = datetime.datetime.now(timezone('US/Eastern')).strftime("%b%d.%Y").lower()

    week_day = datetime.datetime.now(timezone('US/Eastern')).weekday()

    print ("curr day", week_day)


    calendar_data = []
    for back_day in range(-1, 60):
        d = datetime.datetime.now(timezone('US/Eastern')) - datetime.timedelta(days=back_day)

        day_before = d.strftime("%b%d.%Y").lower()
        print (day_before)

        if os.path.isfile(root_dir + "news_data/{}.csv".format(day_before)) == False:
            print ("not found")
            continue

        if os.path.getsize(root_dir + "news_data/{}.csv".format(day_before)) == 0:
            continue

        df = pd.read_csv(root_dir + "news_data/{}.csv".format(day_before))
        calendar_data = [df] + calendar_data

        if len(df) > 0:
            min_time = df["time"].min()
            time_lag_compare = calculate_time_diff(time.time(), min_time)
            print ("time lag", time_lag_compare)
        

    calendar = {"last_check" : time.time(), "day" :  curr_date, "df" : pd.concat(calendar_data)}

    return calendar["df"]


# solve for a and b
def best_fit(X, Y):

    b, a = np.polyfit(X, Y, 1)

    return a, b

def get_features(select_currency, sub_df, after_price_df, y_train, y_train_top, y_train_bottom):

    features = []
    basket_prices = []
    z_scores = []

    prices_map = {}
    for i, compare_pair in enumerate(currency_pairs):

        prices = sub_df["prices" + str(i)].values.tolist()
        end_price = prices[-1]

        basket_prices.append(prices)

        if select_currency not in compare_pair:
            continue


        select_pair = compare_pair
        if select_currency != compare_pair[:3]:
            prices = [1.0 / p for p in prices]
            select_pair = compare_pair[4:7] + "_" + compare_pair[0:3]

        mean_price = np.mean(prices)
        std_price = np.std(prices)
        norm_prices = [(p - mean_price) / std_price for p in prices]

        
        prices_map[select_pair] = end_price
        features.append((prices[-1] - np.mean(prices)) / np.std(prices))
        features.append(np.std(prices))
        features.append(np.mean([a - b for a, b in zip(norm_prices[1:], norm_prices[:-1])]))

        pip_size = 0.0001
        if "JPY" in compare_pair:
            pip_size = 0.01

        if y_train is None:
            continue

        after_prices = after_price_df["prices" + str(i)].values.tolist()
        if select_currency != compare_pair[:3]:
            after_prices = [1.0 / p for p in after_prices]

        if select_pair not in y_train:
            y_train[select_pair] = {}
            y_train_top[select_pair] = {}
            y_train_bottom[select_pair] = {}

        for pip_diff in range(5, 210, 10):
            if pip_diff not in y_train[select_pair]:
                y_train[select_pair][pip_diff] = []
                y_train_top[select_pair][pip_diff] = []
                y_train_bottom[select_pair][pip_diff] = []

            top_barrier = after_prices[0] + (pip_diff * pip_size)
            bottom_barrier = after_prices[0] - (pip_diff * pip_size)
            
            found = False
            for price in after_prices[:24*20]:
                if price >= top_barrier:
                    found = True
                    break

                if price <= bottom_barrier:
                    found = True
                    break

            y_train[select_pair][pip_diff].append(price - after_prices[0])

            '''
            found = False
            for price in after_prices[:24*20]:
                if price >= top_barrier:
                    found = True
                    break

                if price <= after_prices[0] - (50 * pip_size):
                    found = True
                    break

            y_train_top[select_pair][pip_diff].append(price - after_prices[0])

            found = False
            for price in after_prices[:24*20]:
                if price >= after_prices[0] + (50 * pip_size):
                    found = True
                    break

                if price <= bottom_barrier:
                    found = True
                    break

            y_train_bottom[select_pair][pip_diff].append(price - after_prices[0])
            '''

    currency_correlation = {}
    currency_pair_correlation = {}
    for currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:
        correlations = []

        currency_pair_correlation[currency] = {}

        for i, compare_pair1 in enumerate(currency_pairs):

            if currency not in compare_pair1:
                continue

            same_correlations = []
            for j, compare_pair2 in enumerate(currency_pairs):

                if currency not in compare_pair1 or currency not in compare_pair2:
                    continue

                if i == j:
                    continue

                prices1 = basket_prices[i]
                prices2 = basket_prices[j]

                if currency != compare_pair1[:3]:
                    prices1 = [1.0 / p for p in prices1]

                if currency != compare_pair2[:3]:
                    prices2 = [1.0 / p for p in prices2]
             
                correlation, p_value = stats.pearsonr(prices1, prices2)
                same_correlations.append(correlation)

            currency_pair_correlation[currency][compare_pair1] = np.mean(same_correlations)


            for j, compare_pair2 in enumerate(currency_pairs):

                if i <= j:
                    continue

                if currency not in compare_pair1 or currency not in compare_pair2:
                    continue

                prices1 = basket_prices[i]
                prices2 = basket_prices[j]

                if currency != compare_pair1[:3]:
                    prices1 = [1.0 / p for p in prices1]

                if currency != compare_pair2[:3]:
                    prices2 = [1.0 / p for p in prices2]
             
                correlation, p_value = stats.pearsonr(prices1, prices2)
                correlations.append(correlation)

        currency_correlation[currency] = np.mean(correlations)

    features += [currency_correlation, currency_pair_correlation, sub_df.tail(1)["times"].values.tolist()[-1], prices_map]

    return features

def create_correlation_graph(select_currency):

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
    

    X_train = []
    y_train = {}
    y_train_top = {}
    y_train_bottom = {}
    for index in range(1, len(times), 24):
        print (index, len(times))
        sub_df = before_all_price_df[before_all_price_df["times"] <= times[index]].tail(24 * 20)
        after_price_df = before_all_price_df[before_all_price_df["times"] >= times[index]]
        if len(after_price_df) <= 24 * 20:
            continue

        features = get_features(select_currency, sub_df, after_price_df, y_train, y_train_top, y_train_bottom)

        X_train.append(features)

    return X_train, y_train, y_train_top, y_train_bottom

def make_forecast(select_currency):

    X_train, y_train, y_train_top, y_train_bottom = create_correlation_graph(select_currency)
    
    count = 0
    for i, compare_pair in enumerate(currency_pairs):
        if select_currency not in compare_pair:
            continue

        prices, times = get_time_series(compare_pair, 24 * 30)
        before_price_df2 = pd.DataFrame()
        before_price_df2["prices" + str(i)] = prices
        before_price_df2["times"] = times
        count += 1

        if count > 1:
            before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
            before_all_price_df.reset_index(inplace=True)
        else:
            before_all_price_df = before_price_df2

    sub_df = before_all_price_df.tail(24 * 20)
    features = get_features(select_currency, sub_df, None, None)

    kwargs = dict(hist_kws={'alpha':0.6}, kde_kws={'linewidth':2}, hist=False)
    plt.figure(figsize=(5,4))


    plt.title("Pip Movement Basket - {} Pairs".format(select_currency))

    for pair in y_train:
        print (pair)

        pip_dist = []
        for pip_diff in y_train[pair]:

            clf = xgb.XGBClassifier()
            clf.fit(np.array(X_train), np.array(y_train[pair][pip_diff]))

            prob = clf.predict_proba(np.array([features]))[0][1]

            if prob >= 0.5:
                pip_dist += [pip_diff] * int(prob * 100)
                pip_dist += [-pip_diff] * int((1 - prob) * 100)
            else:
                pip_dist += [-pip_diff] * int((1 - prob) * 100)
                pip_dist += [pip_diff] * int(prob * 100)

        right = len([p for p in pip_dist if p > 0])
        left = len([p for p in pip_dist if p < 0])
        print (right, left)

        if right > left:
            prob = int(100 * (float(right) / (right + left)))
        else:
            prob = -int(100 * (float(left) / (right + left)))

        if prob > 0:
            sns.distplot(pip_dist, label="BUY {} {}%".format(pair, prob), **kwargs)
        else:
            sns.distplot(pip_dist, label="SELL {} {}%".format(pair, prob), **kwargs)

    plt.axvline(0, color='black') 
    plt.xlabel("Pip Movement")
    plt.legend()
    plt.savefig("/var/www/html/images/{}_basket_stat_arb_forecast.png".format(select_currency))
    plt.close()
    #plt.show()

class Order():

    def __init__(self):
        self.amount = 0

def trade_martingale(select_currency, is_alt):

    if os.path.isfile("X1_{}.pickle".format(select_currency)) == False:
        X, y, _, _ = create_correlation_graph(select_currency)

        pickle.dump(X, open("X1_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y, open("y1_{}.pickle".format(select_currency), "wb"))
    

    X = pickle.load(open("X1_{}.pickle".format(select_currency), "rb"))
    y = pickle.load(open("y1_{}.pickle".format(select_currency), "rb"))


    pair_subset = []
    pair_index = {}
    for i, compare_pair in enumerate(currency_pairs):
        if select_currency in compare_pair:

            if select_currency != compare_pair[:3]:
                select_pair = compare_pair[4:7] + "_" + compare_pair[0:3]
            else:
                select_pair = compare_pair

            pair_index[select_pair] = len(pair_subset) * 3
            pair_subset.append(select_pair)

    orders = []
    equity = 5000
    max_orders = 0
    marigns = []
    equity_curve = []
    order_counts = []
    order_pnls = []

    for pip_diff in [55]:

        pip_size = 0.0001
  
        for i, x_sample in zip(range(len(X)), X):

            price_map = x_sample[-1]
            timestamp = x_sample[-2]

            correlation_pair = x_sample[-3][select_currency]
            currency_correlation = x_sample[-4]

            correlation = currency_correlation[select_currency]
            mean_correlations = np.mean(currency_correlation.values())

            z_diffs = []
            grads = []
            for sub_pair in pair_subset:

                if sub_pair not in currency_pairs:
                    select_pair = sub_pair[4:7] + "_" + sub_pair[0:3]
                else:
                    select_pair = sub_pair


                if is_alt:
                    z_diffs.append([sub_pair, x_sample[pair_index[sub_pair]], correlation_pair[select_pair]])
                else:
                    z_diffs.append([sub_pair, x_sample[pair_index[sub_pair]], correlation_pair[select_pair]])

                grads.append(x_sample[pair_index[sub_pair] + 2])

            mean_z_score = np.mean([a[1] for a in z_diffs])
            z_diffs = [[a[0], a[1], a[2], a[1] - mean_z_score] for a in z_diffs]
            z_diffs = sorted(z_diffs, key=lambda x: x[1])
            mean_grad = np.mean(grads)

            before = len(orders)
            if correlation > 0:

                order = Order()
                order.open_price = price_map[z_diffs[0][0]]

                order.pair = z_diffs[0][0]

                order.dir = ((order.pair in currency_pairs) == False)
                order.z_score = z_diffs[0][1] < mean_z_score
                curr_dir = order.dir

                if is_alt:
                    order.amount = (5000 / 5000) * 5000 * z_diffs[0][2] * 2
                else:
                    order.amount = (5000 / 5000) * 5000 * z_diffs[0][2] * 2 
                
                if abs(x_sample[pair_index[order.pair]] - mean_z_score) > 0:
                    if abs(z_diffs[0][1]) < abs(z_diffs[-1][1]):
                        orders.append(order)    

                order = Order()
                order.open_price = price_map[z_diffs[-1][0]]
                order.pair = z_diffs[-1][0] 
                order.z_score = z_diffs[-1][1] > mean_z_score
                order.dir = ((order.pair in currency_pairs) == True)

                if is_alt:
                    order.amount = (5000 / 5000) * 5000 * z_diffs[-1][2] * 2
                else:
                    order.amount = (5000 / 5000) * 5000 * z_diffs[-1][2] * 2 

                if abs(x_sample[pair_index[order.pair]] - mean_z_score) > 0:
                    if abs(z_diffs[0][1]) > abs(z_diffs[-1][1]):
                        orders.append(order)

            print ("increase in orders", len(orders) - before)

            total_pnl = 0
            total_order_amount = 0
            new_orders = []
            for order in orders:

                if "JPY" in order.pair:
                    pip_size = 0.01
                else:
                    pip_size = 0.0001

                if order.dir == (price_map[order.pair] > order.open_price):
                    order.pnl = abs(price_map[order.pair] - order.open_price)
                else:
                    order.pnl = -abs(price_map[order.pair] - order.open_price)

                amount = order.amount

                total_order_amount += amount

                order.pnl /= pip_size
                order.pnl -= 5

                if order.pnl < -100:
                    order.pnl *= amount
                    equity += order.pnl
                    order_pnls.append(order.pnl )
                    print ("loss ---------------------------")
                    continue

                if order.z_score != (x_sample[pair_index[order.pair]] < mean_z_score) and is_alt == True:
                    order.pnl *= amount
                    equity += order.pnl
                    order_pnls.append(order.pnl )
                    print ("profit")
                    continue

                '''
                print ("variance", order.pair, (x_sample[pair_index[order.pair] + 1] / pip_size))
                if order.pnl < -200 / (total_order_amount / (equity + total_pnl)) and is_alt == True:
                    order.pnl *= amount
                    equity += order.pnl
                    print ("loss")
                    continue  
                '''
                
                order.pnl *= amount
                total_pnl += order.pnl

                new_orders.append(order)

            orders = new_orders

            marigns.append(total_order_amount / (equity + total_pnl))

            if total_pnl > 0 and is_alt == False:
                order_pnls += ([order.pnl for order in orders])
                equity += total_pnl
                orders = []


            print (equity + total_pnl, len(orders), select_currency)
            max_orders = max(max_orders, len(orders))
            equity_curve.append(equity + total_pnl)
            order_counts.append(len(orders))

    returns = [a - b for a, b in zip(equity_curve[1:], equity_curve[:-1])]
    print ("max orders", max_orders)
    print ("max margin", (max(marigns) / 50) * 100)
    print ("Sharpe", np.mean(returns) / np.std(returns))

    return np.mean(returns) / np.std(returns), (max(marigns) / 50) * 100, np.sum(returns), (np.mean(marigns) / 50) * 100
 

def test_model(select_currency, train_index):

    if os.path.isfile("X_{}.pickle".format(select_currency)) == False:
        X, y, y_top, y_bottom = create_correlation_graph(select_currency)

        pickle.dump(X, open("X_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y, open("y_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y_top, open("y_top_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y_bottom, open("y_bottom_{}.pickle".format(select_currency), "wb"))
    

    X = pickle.load(open("X_{}.pickle".format(select_currency), "rb"))
    y = pickle.load(open("y_{}.pickle".format(select_currency), "rb"))
    y_top = pickle.load(open("y_top_{}.pickle".format(select_currency), "rb"))
    y_bottom = pickle.load(open("y_bottom_{}.pickle".format(select_currency), "rb"))

    if train_index >= len(X):
        return None, None

    print (len(X))

    pair_subset = []
    pair_index = {}
    for i, compare_pair in enumerate(currency_pairs):
        if select_currency in compare_pair:

            if select_currency != compare_pair[:3]:
                select_pair = compare_pair[4:7] + "_" + compare_pair[0:3]
            else:
                select_pair = compare_pair

            pair_index[select_pair] = len(pair_subset)
            pair_subset.append(select_pair)

    auc_pair = {}

    select_params = []
    profit_this_year = 0


    for pair in y:

        if "JPY" in pair:
            continue


        for pip_diff in [145]:


            for is_reverse in [True, False]:
                for corr_threshold in [0.4, 0.6, 0.8, 0.9]:
                    for z_treshold in [1.4, 1.6, 1.8, 2.0]:

                        #print (corr_threshold, z_treshold, is_reverse)

                        profits = {}
                        if pip_diff not in profits:
                            profits[pip_diff] = []

                        pip_size = 0.0001
                  
                        train_set = range(train_index) + range(train_index + 250, len(X))

                        for x_sample, y_sample in zip(X, y[pair][pip_diff]):
                            pip_range = abs(y_sample / pip_size)

                            if i not in train_set:
                                continue

                            z_diffs = []
                            for sub_pair in pair_subset:
                                if sub_pair != pair:
                                    z_diffs.append(x_sample[pair_index[pair]] - x_sample[pair_index[sub_pair]])

                     
                            timestamp = x_sample[-1]
                            currency_correlation = x_sample[-2]
                            correlation = currency_correlation[pair[:3]]
                            compare_correlation = currency_correlation[pair[4:7]]

              
                            if correlation > corr_threshold and abs(np.mean(z_diffs)) > z_treshold:

                                if is_reverse:
                                 
                                    if (np.mean(z_diffs) > 0) == (y_sample < 0):
                                        profits[pip_diff].append((pip_range - 5) )
                                    else:
                                        profits[pip_diff].append((-pip_range - 5) )
                                else:
                                   
                                    if (np.mean(z_diffs) > 0) == (y_sample > 0):
                                        profits[pip_diff].append((pip_range - 5) )
                                    else:
                                        profits[pip_diff].append((-pip_range - 5) )

                        if sum(profits[pip_diff]) > 0:
                            
                            sharpe = np.mean(profits[pip_diff]) / np.std(profits[pip_diff])

                            if sharpe >= 0.20 and len(profits[pip_diff]) > 120:
                                
                                profits[pip_diff] = []
                                for i, x_sample, y_sample in zip(range(len(X)), X, y[pair][pip_diff]):
                                    pip_range = abs(y_sample / pips_size)

                                    if i in train_set:
                                        continue

                                    z_diffs = []
                                    for sub_pair in pair_subset:
                                        if sub_pair != pair:
                                            z_diffs.append(x_sample[pair_index[pair]] - x_sample[pair_index[sub_pair]])

                                    currency_correlation = x_sample[-2]
                                    correlation = currency_correlation[pair[:3]]
                                    compare_correlation = currency_correlation[pair[4:7]]

                                    if  correlation > corr_threshold and abs(np.mean(z_diffs)) > z_treshold:

                                        if is_reverse:
                                            if (np.mean(z_diffs) > 0) == (y_sample < 0):
                                                profits[pip_diff].append((pip_range - 5) )
                                            else:
                                                profits[pip_diff].append((-pip_range - 5) )
                                        else:
                    
                                            if (np.mean(z_diffs) > 0) == (y_sample > 0):
                                                profits[pip_diff].append((pip_range - 5) )
                                            else:
                                                profits[pip_diff].append((-pip_range - 5) )
                               
                                if len(profits[pip_diff]) > 0:
                                    print (pair, pip_diff, np.mean(profits[pip_diff]) / np.std(profits[pip_diff]), sum(profits[pip_diff]), len(profits[pip_diff]))
                                    profit_this_year += sum(profits[pip_diff])

                                select_params.append({
                                    "select_currency" : select_currency,
                                    "sharpe": sharpe,
                                    "is_reverse" : is_reverse,
                                    "correlation_threshold" : corr_threshold,
                                    "z_diff_threshold" : z_treshold,
                                    "pip_diff" : pip_diff,
                                    "pair" : pair
                                    })


    return select_params, profit_this_year

def get_volume_diff(timestamp, price_df):

    sub_df = price_df[price_df["times"] <= timestamp].tail(24 * 8)

    volumes = sub_df["volumes"].values.tolist()
    highs = sub_df["highs"].values.tolist()
    lows = sub_df["lows"].values.tolist()
    opens = sub_df["opens"].values.tolist()

    v_diffs = []
    for l, h, v, o in zip(lows, highs, volumes, opens):

        if abs(h - l) > 0:
            ratio1 = (abs(l - o) / (h - l)) * v
            ratio2 = (abs(h - o) / (h - l)) * v

            v_diffs.append(ratio1 - ratio2)

    v_sum = []
    for index in range(0, len(v_diffs), 24):
        v_sum.append(sum(v_diffs[index:index + 24]))

    return v_sum


def model_approach(select_currency, profit_by_year):

    if os.path.isfile("y_top_{}.pickle".format(select_currency)) == False:
        X, y, y_top, y_bottom = create_correlation_graph(select_currency)

        pickle.dump(X, open("X_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y, open("y_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y_top, open("y_top_{}.pickle".format(select_currency), "wb"))
        pickle.dump(y_bottom, open("y_bottom_{}.pickle".format(select_currency), "wb"))
    

    X = pickle.load(open("X_{}.pickle".format(select_currency), "rb"))
    y = pickle.load(open("y_{}.pickle".format(select_currency), "rb"))
    y_top = pickle.load(open("y_top_{}.pickle".format(select_currency), "rb"))
    y_bottom = pickle.load(open("y_bottom_{}.pickle".format(select_currency), "rb"))

    print (len(X))

    pair_subset = []
    pair_index = {}
    for i, compare_pair in enumerate(currency_pairs):
        if select_currency in compare_pair:

            if select_currency != compare_pair[:3]:
                select_pair = compare_pair[4:7] + "_" + compare_pair[0:3]
            else:
                select_pair = compare_pair

            pair_index[select_pair] = len(pair_subset)
            pair_subset.append(select_pair)

    select_params = []

    vol_pairs = ["AUD_CAD", "AUD_CHF", "AUD_NZD", "AUD_USD"]
    
    sharpe_diffs = []

    for pair in y:

        if "JPY" in pair:
            continue

        '''
        if pair not in vol_pairs:
            continue
        '''

        select_pair = pair
        if pair not in currency_pairs:
            select_pair = pair[4:7] + "_" + pair[:3]

        prices, times, volumes, lows, highs = load_time_series(select_pair, None, False)
        before_price_df2 = pd.DataFrame()
        before_price_df2["opens"] = prices
        before_price_df2["lows"] = lows
        before_price_df2["highs"] = highs
        before_price_df2["volumes"] = volumes
        before_price_df2["times"] = times

        #df = pd.read_excel("/Users/andrewstevens/Downloads/AUD_Crosses_MarketForceData.xlsx", pair.replace("_", ""))
        df = pd.read_excel("/Users/andrewstevens/Downloads/AUD_Crosses_MarketForceData.xlsx", "AUDCAD")

        #gmt = est - 5
        df["Date"] = df["Date"].apply(lambda x: calendar.timegm(datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timetuple()))
        df["Date"] = df["Date"].apply(lambda x: x + (5 * 60 * 60))
        df["Diff"] = df.apply(lambda x: x["VOL_Long%"] - x["Vol_Short%"], axis=1)
        dates = df["Date"].values.tolist()

        mean_sharpe_map = {}
        for feature_set in [6]:
            X_sub = []
            y_sub_top = []
            y_sub_bottom = []
            y_sub = []
            for pip_diff in [145]:

                import numpy as np
                for x_sample, y_sample, y_sample_top, y_sample_bottom in zip(X, y[pair][pip_diff], y_top[pair][pip_diff], y_bottom[pair][pip_diff]):
                    pip_range = abs(y_sample)

                    z_diffs = []
                    for sub_pair in pair_subset:
                        if sub_pair != pair:
                            z_diffs.append(x_sample[pair_index[pair]] - x_sample[pair_index[sub_pair]])

                    price_map = x_sample[-1]
                    timestamp = x_sample[-2]
                    currency_correlation = x_sample[-3]
          
                    correlation = currency_correlation[pair[:3]]
                    compare_correlation = currency_correlation[pair[4:7]]

                    '''
                    sub_df = df[df["Date"] <= timestamp]
                    diffs = sub_df.tail(5)["Diff"].values.tolist()

                    if len(diffs) < 5:
                        continue
                    '''

                    diffs1 = get_volume_diff(timestamp, before_price_df2)
                    vol_dffs = [a - diffs1[-1] for a in diffs1[:-1]]
            
                    correlations = [currency_correlation[currency] for currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]]

                    if feature_set == 1:
                        X_sub.append([correlation - compare_correlation] + z_diffs + diffs + vol_dffs)
                    elif feature_set == 2:
                        X_sub.append([correlation, np.percentile(z_diffs, 50), compare_correlation] + vol_dffs)
                    elif feature_set == 3:
                        X_sub.append([np.percentile(z_diffs, 50), correlation - compare_correlation] + vol_dffs)
                    elif feature_set == 4:
                        X_sub.append([correlation - compare_correlation] + z_diffs + vol_dffs)
                    elif feature_set == 5:
                        X_sub.append([compare_correlation, correlation, correlation - compare_correlation, np.mean(z_diffs)] + vol_dffs)
                    elif feature_set == 6:
                        X_sub.append([correlation - compare_correlation] + [np.percentile(z_diffs, 50), np.mean(z_diffs)] + z_diffs + vol_dffs)

                    if len(X_sub[-1]) == 16:
                        y_sub.append(y_sample)
                        y_sub_top.append(y_sample_top)
                        y_sub_bottom.append(y_sample_bottom)


            import numpy as np
            import matplotlib.pyplot as plt

            y_final = [1 if y1 > 0 else -1 for y1 in y_sub]

            temp_profits_by_year = {}
            Y_actuals = []
            Y_preds = []
            for index in range(0, len(X_sub), 100):
                X_train = X_sub[:index] + X_sub[index + 200:]
                y_train = y_final[:index] + y_final[index + 200:]

                X_test = X_sub[index+20:index + 200 - 20]
                y_test = y_final[index+20:index + 200 - 20]

        

                try:
                    clf = xgb.XGBRegressor(n_tress = 1000)
                    clf.fit(np.array(X_train), y_train)
                except:
                    continue

                '''
                if feature_set == 1:
                    feaure_names = ["correlation", "compare_correlation"] + (len(z_diffs) * ["z_diff"]) + (len(diffs) * ["Market Force"]) + (len(vol_dffs) * ["Vol Diffs"])
                    print (len(feaure_names))
                    explainer = shap.TreeExplainer(clf)
                    X = pd.DataFrame(X_train, columns=feaure_names)
                    shap_values = explainer.shap_values(X)
                    shap.summary_plot(shap_values, X, plot_type="bar")
                '''

                predictions = clf.predict(np.array(X_train))
                predictions = [abs(p) for p in predictions]
                threshold = np.percentile(predictions, 80)

                try:
                    probs = clf.predict(np.array(X_test))
                except:
                    continue

                pip_size = 0.0001

                if index not in temp_profits_by_year:
                    temp_profits_by_year[index] = []

                Y_actuals += list(y_test)
                Y_preds += list(probs)

                
                for y1, y_top1, y_bottom1, p1 in zip(y_sub[index+20:index + 200 - 20], y_sub_top[index+20:index + 200 - 20], y_sub_bottom[index+20:index + 200 - 20], probs):

                    if abs(p1) >= threshold:

                        '''
                        if p1 > 0:
                            y1 = y_top1
                        else:
                            y1 = y_bottom1
                        '''

                        if (y1 >= 0) == (p1 >= 0):
                            temp_profits_by_year[index].append((abs(y1 / pip_size)-5) * abs(p1))
                        else:
                            temp_profits_by_year[index].append((-abs(y1 / pip_size)-5) * abs(p1))


            Y_test = Y_actuals
            Y_pred = Y_preds

            '''
            r_squared = 0.59
            plt.scatter(Y_test,Y_pred)
            plt.xlabel('Actual values')s

            plt.ylabel('Predicted values')

            plt.plot(np.unique(Y_test), np.poly1d(np.polyfit(Y_test, Y_pred, 1))(np.unique(Y_test)))

            plt.text(0.6, 0.5, 'R-squared = %0.2f' % r_squared)
            plt.show()
            '''

            sharpes = []
            print ("feature_set", feature_set, len(X_sub))
            for index in range(0, len(X_sub), 100):

                profits = []
                for i in range(0, len(X_sub), 100):
                    if i != index:
                        profits += temp_profits_by_year[i]

                if index not in profit_by_year[feature_set]:
                    profit_by_year[feature_set][index] = 0

                
                sharpe = np.mean(profits) / np.std(profits)
                
                #print (pair, "Sharpe", np.mean(profits) / np.std(profits), len(profits)) 

                for compare_sharpe in [0.2]:
                    if sharpe >= compare_sharpe:
                        profit_by_year[feature_set][index] += sum([v for v in temp_profits_by_year[index]])
                     

            mean_sharpe_map[feature_set] = np.mean(sharpes)

            #print ("Mean Sharpe", np.mean(sharpes))

        sharpe1 = np.mean(profit_by_year[1].values()) / np.std(np.mean(profit_by_year[1].values()))
        sharpe2 = np.mean(profit_by_year[6].values()) / np.std(np.mean(profit_by_year[6].values()))
        print ("Mean Sharpe Diff (With - Without Market Force)", round((sharpe1 - sharpe) * 100, 2), "%")
        print ("With Market Force Features Yearly Profit", profit_by_year[1])
        print ("Without Market Force Features Yearly Profit", profit_by_year[6])

        print ("Combined", [profit_by_year[1][key] + profit_by_year[6][key] for key in profit_by_year[1]])

      
    return profit_by_year

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) 
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) 
    return std, returns

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

    return -p_ret / p_std

def max_sharpe_ratio(returns, optimization_bounds = (0.0, 1.0)):

    df = pd.DataFrame(returns)
    mean_returns = df.mean()
    cov_matrix = df.cov()

    num_assets = len(mean_returns)
    avg_weight = 1.0 / num_assets

   # constraints = [{'type': 'ineq', 'fun': lambda x: +x[i] - avg_weight * max_exposure} for i in range(num_assets)]
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
    args = (mean_returns, cov_matrix, 0)
    bound = optimization_bounds
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints, options={"maxiter":10})

    weights = result['x'].tolist() 
    return weights

def optimize_portfolio():
    profit_by_year = pickle.load(open("profit_by_year.pickle", "rb"))

    keys = sorted(profit_by_year.keys())

    X_train = []
    for key in keys:
        returns = []

        features = sorted(profit_by_year[key].keys())
        for f in features:
            returns.append(profit_by_year[key][f])

        X_train.append(returns)

    select_columns = []
    df = pd.DataFrame(X_train, columns = features)
    for column in df.columns:
        if df[column].sum() > 0:
            select_columns.append(column)

    X_train = df[select_columns].values.tolist()

    for index in range(len(X_train)):
        X = X_train[:index-1] + X_train[index + 2:]

        weights = max_sharpe_ratio(X)
        #print (weights)

        #print (sum(X_train[index]))

        print (sum([x * w for w, x in zip(weights, X_train[index])]))
        
profit_by_year = {}
profit_by_year[1] = {}
profit_by_year[6] = {}

sharpe_diffs = []
margin_diffs = []
return_diffs = []
order_margin_diffs = []

sharpe1_mean = []
sharpe2_mean = []
for select_currency in ["NZD", "AUD", "USD", "CAD", "GBP", "EUR", "CHF"]:
    sharpe1, margin1, return1, order_margin1 = trade_martingale(select_currency, True)

    sharpe2, margin2, return2, order_margin2 = trade_martingale(select_currency, True)

    sharpe_diffs.append(sharpe2 - sharpe1)
    margin_diffs.append((margin2 - margin1) / max([margin2, margin1]))
    return_diffs.append((return2 - return1) / max([return1, return2]))
    order_margin_diffs.append(order_margin2 - order_margin1)

    sharpe1_mean.append(sharpe1)
    sharpe2_mean.append(sharpe2)


print ("mean sharpe diff", np.mean(sharpe_diffs))
print ("mean margins diff", np.mean(margin_diffs))
print ("mean return diff", np.mean(return_diffs))
print ("mean order margin diff", np.mean(order_margin_diffs))

print ("sharpe1", np.mean(sharpe1_mean))
print ("sharpe2", np.mean(sharpe2_mean))

sys.exit(0)


for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:
    profit_by_year = model_approach(select_currency, profit_by_year)

    keys = sorted(profit_by_year.keys())

    for key in keys:
        print (key, profit_by_year[key])

    pickle.dump(profit_by_year, open("profit_by_year.pickle", "wb"))

optimize_portfolio()

sys.exit(0)

offset = 0
for i in range(200):
    select_params_map = {}
    profit_this_year = 0
    profit_by_currency = {}
    for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:
        select_params, profit = test_model(select_currency, offset)
        profit_this_year += profit 

        profit_by_currency[select_currency] = profit

        select_params_map[select_currency] = len(select_params)

    offset += 100
    print ("total_profit_this_year", profit_this_year)
    print ("profit by currency")
    print (profit_by_currency)


checkIfProcessRunning('stat_arb_price_distribution.py', "")

trade_logger = setup_logger('first_logger', root_dir + "stat_arb_price_forecast.log")
trade_logger.info('Starting ') 


try: 
    for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:
        trade_logger.info(select_currency) 
        trade_martingale(select_currency)

    trade_logger.info('Finished ') 
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())




