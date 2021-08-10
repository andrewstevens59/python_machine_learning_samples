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
    z_score_map = {}
    price_delta_map = {}
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
        z_score_map[select_pair] = (prices[-1] - np.mean(prices)) / np.std(prices)
        price_delta_map[select_pair] = np.mean([a - b for a, b in zip(norm_prices[1:], norm_prices[:-1])])


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

    features = [z_score_map, price_delta_map, currency_correlation, currency_pair_correlation, sub_df.tail(1)["times"].values.tolist()[-1], prices_map]

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
        time_frame_features = {}
        for time_frame in [10, 5, 2, 20]:
            sub_df = before_all_price_df[before_all_price_df["times"] <= times[index]].tail(24 * time_frame)
  
            time_frame_features[time_frame] = get_features(select_currency, sub_df, None, y_train, y_train_top, y_train_bottom)

        X_train.append(time_frame_features)

    return X_train



class Order():

    def __init__(self):
        self.amount = 0

def caclulate_profit_curve(select_currency, transition_mat, profit_transition, cluster_num):

    for i in range(cluster_num):
        for j in range(cluster_num):

            if transition_mat[i][j] > 0:
                profit_transition[i][j] /= transition_mat[i][j]

    for i in range(cluster_num):

        net_sum = 0
        for j in range(cluster_num):
            net_sum += transition_mat[i][j] 

        if net_sum > 0:
            for j in range(cluster_num):
                transition_mat[i][j] /= net_sum


    import matplotlib.pyplot as plt

    forecasted_profit = {}
    for node in range(cluster_num):

        print (node, "-------------")
        pdf = np.zeros((1, cluster_num))
        pdf[0][node] = 1.0

        profits = []
        curr_profit = 0
        for iteration in range(cluster_num):

            net_profit = 0
            for i in range(cluster_num):
                for j in range(cluster_num): 
                    net_profit += profit_transition[i][j] * pdf[0][i] * transition_mat[i][j]


            pdf = np.matmul(pdf, transition_mat)
            curr_profit += net_profit
            profits.append(curr_profit)

        net_sum = 0
        for i in range(cluster_num):
            net_sum += pdf[0][i]

        forecasted_profit[node] = profits[-1]

        #plt.plot(range(len(profits)), profits, label = "Node: {}".format(node))

    '''
    plt.legend()
    plt.show()
    sys.exit(0)
    '''

    return forecasted_profit

def trade_momentum(select_currency, is_alt, profit_by_segment_map, time_frame):


    if os.path.isfile("X_momentum6_{}.pickle".format(select_currency)) == False:
        X = create_correlation_graph(select_currency)

        pickle.dump(X, open("X_momentum6_{}.pickle".format(select_currency), "wb"))
    
    X = pickle.load(open("X_momentum6_{}.pickle".format(select_currency), "rb"))

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

    sell_num = 0
    buy_num = 0
    neg_profits_count = 0
    pos_profits_count = 0

    states = []
    open_prices = []

    pip_size = 0.0001

    for i, x_sample in zip(range(len(X)), X):

        if time_frame not in x_sample:
            continue

        z_score_map = x_sample[time_frame][0]
        price_delta_map = x_sample[time_frame][1]

        price_map = x_sample[time_frame][-1]
        timestamp = x_sample[time_frame][-2]

        correlation_pair = x_sample[time_frame][-3][select_currency]
        currency_correlation = x_sample[time_frame][-4]

        correlation = currency_correlation[select_currency]
        mean_correlations = np.mean(currency_correlation.values())
        mean_z_score = np.mean(z_score_map.values())

        print ("z_score", mean_z_score, "buy_num", buy_num, "sell_num", sell_num)

        z_diffs = []
        grads = []
        z_scores = []
        for sub_pair in pair_subset:

            if "JPY" in sub_pair:
                continue

            if sub_pair not in currency_pairs:
                select_pair = sub_pair[4:7] + "_" + sub_pair[0:3]
            else:
                select_pair = sub_pair

            z_score = z_score_map[sub_pair]
            price_delta = price_delta_map[sub_pair]

            z_diffs.append([sub_pair, z_score])
            z_scores.append(z_score)

        z_score_mean = np.mean(z_scores)
        z_diffs = sorted(z_diffs, key=lambda x: x[1])


        states.append([correlation, z_diffs[-1][1] - z_diffs[0][1]])

        open_prices.append([price_map[z_diffs[-1][0]], price_map[z_diffs[0][0]], abs(z_diffs[-1][1] - z_score_mean), abs(z_diffs[0][1] - z_score_mean)])

    equity = 0
    cluster_num = 3
    for test_index in range(1, len(states), 120):

  
        train_states = states[:test_index] + states[test_index + 200:]
        train_open_prices = open_prices[:test_index] + open_prices[test_index + 200:]

        test_states = states[test_index+2:test_index + 200 - 2]
        test_open_prices = open_prices[test_index+2:test_index + 200 - 2]

        kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(train_states)
        clusters = kmeans.predict(train_states)

        transition_mat = np.zeros((cluster_num, cluster_num))
        profit_transition = np.zeros((cluster_num, cluster_num))

        for cluster_id in range(len(clusters) - 1):

            cluster_i = clusters[cluster_id]
            cluster_j = clusters[cluster_id+1]

            transition_mat[cluster_i][cluster_j] += 1

            pnl1 = (train_open_prices[cluster_id+1][1] - train_open_prices[cluster_id][1])

            pnl2 = (-train_open_prices[cluster_id+1][0] + train_open_prices[cluster_id][0]) 

            profit_transition[cluster_i][cluster_j] += -pnl1 - pnl2

        print (transition_mat)
        print (profit_transition)

        forecasted_profit = caclulate_profit_curve(select_currency, transition_mat, profit_transition, cluster_num)
        max_value = max(forecasted_profit.values())

        cluster_ids = kmeans.predict(test_states)

        total_profit = 0
        previous_cluster_id =  None
        for cluster_id in range(len(cluster_ids) - 1):

            pnl1 = (test_open_prices[cluster_id+1][1] - test_open_prices[cluster_id][1])
 
            pnl2 = (-test_open_prices[cluster_id+1][0] + test_open_prices[cluster_id][0]) 

            cluster_i = cluster_ids[cluster_id]

            if forecasted_profit[cluster_i] == max_value:
                total_profit += -pnl1 - pnl2

            if cluster_i != previous_cluster_id:
                total_profit -= 0.001
                previous_cluster_id = cluster_i

        if test_index not in profit_by_segment_map:
            profit_by_segment_map[test_index] = 0

        profit_by_segment_map[test_index] += total_profit

        equity += total_profit
        print ("equity", equity, total_profit)

    return equity

total_profit = 0
profit_by_segment_map = {}

for time_frame in [2]:
    for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR"]:

        try:
            total_profit += trade_momentum(select_currency, True, profit_by_segment_map, time_frame)
        except:
            pass


print ("Total Profit", total_profit)
print (profit_by_segment_map)





