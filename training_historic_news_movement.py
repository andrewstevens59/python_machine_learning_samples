import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle
import xgboost as xgb
from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from bayes_opt import BayesianOptimization
from datetime import timedelta
from sklearn import metrics
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import gzip, cPickle
import string
import random as rand

import os
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from uuid import getnode as get_mac
import socket
import paramiko
import json
import enum

import os

import mysql.connector

all_currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


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

            if high != low:
                prices.append(c_price)
                times.append(time)
                volumes.append(volume)

    return prices, times, volumes

class Order:

    def __init__(self):
        self.pair = ""
        self.dir = 0
        self.open_price = 0
        self.time = 0
        self.readable_time = ""
        self.amount = 0
        self.id = 0
        self.side = 0
        self.pnl = 0
        self.max_pnl = 0
        self.open_predict = 0
        self.tp_price = 0
        self.sl_price = 0
        self.hold_time = 0
        self.is_invert = False
        self.invert_num = 0
        self.reduce_amount = 0
        self.match_amount = 0
        self.equity_factor = 0


def calculate_time_diff(now_time, ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    date = datetime.datetime.utcfromtimestamp(now_time).strftime('%Y-%m-%d %H:%M:%S')
    e = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    def helper(d):
        if d.weekday() == 5:
            d += timedelta(days=1)
        return d.replace(hour=0, minute=0, second=0, microsecond=0)


    if e.weekday() in {5, 6}:
        e = helper(e)
    if s.weekday() in {5, 6}:
        s = helper(s)
    _diff = (e - s)
    while s < e:
        if s.weekday() in {5, 6}:
            _diff -= timedelta(days=1)
        elif s.weekday() == 0:
            s += timedelta(days=4)
        s += timedelta(days=1)

    return (_diff.total_seconds() / (60 * 60))



class ModelType(enum.Enum): 
    barrier = 1
    time_regression = 2
    time_classification = 3

def back_test(select_pair, price_df, setup_rows):


    prev_releases = []
    curr_release_time = 0

    min_time = 99999999999999999
    max_time = 0
    prev_time_stamp = 0

    X_train = []
    y_train = []
    times = []

    for row in setup_rows:
        currency = row[0]
        currency_pair = row[1]
        time_stamp = row[2]
        release_time = row[3]

        barrier = row[4]
        prob = row[5]
        auc = row[6]

        exchange_rate = row[9]

        min_time = min(min_time, time_stamp)
        max_time = max(max_time, time_stamp)

        if time_stamp != curr_release_time:

            curr_release_time = time_stamp
            future_prices = price_df["prices"][(price_df["times"] >= prev_time_stamp)].values.tolist()
            #print (prev_time_stamp, curr_release_time)

            prev_time_stamp = curr_release_time

            X_last = []
            auc_barrier_mult = 0
            for select_currency in ["EUR", "JPY"]:
                frequency = [0] * 10
                for auc_barrier_mult_index in range(0, 10):
                    auc_barrier_mult += 0.25
                    for release in prev_releases:
                        time_stamp = release[2]
                        relase_time = release[3]
                        currency = release[0]
                        barrier = release[4]
                        prob = release[5]
                        auc = release[6]

                        if currency != select_currency:
                            continue

                        if abs(prob - 0.5) < 0.5 - max(0, (auc - 0.5) * auc_barrier_mult):
                            continue

                        if prob > 0.5:
                            frequency[auc_barrier_mult_index] += 1
                        else:
                            frequency[auc_barrier_mult_index] -= 1

                X_last += frequency

            if select_pair[4:7] == "JPY":
                pip_size = 0.01
            else:
                pip_size = 0.0001

            pip_size *= 5

            y_barrier = [None] * 21
            for barrier_index in range(1, 21):

                start_price = future_prices[0]
                top_barrier = start_price + (pip_size + (pip_size * barrier_index))
                bottom_barrier = start_price - (pip_size + (pip_size * barrier_index))

                for price in future_prices:
                    
                    if price >= top_barrier:
                        y_barrier[barrier_index] = True
                        break

                    if price <= bottom_barrier:
                        y_barrier[barrier_index] = False
                        break

            X_train.append(X_last)
            y_train.append(y_barrier)
            times.append(time_stamp)

            prev_releases = []

        prev_releases.append(row)

    return X_train, y_train, times

def create_training_set(select_pair):

    print (select_pair)

    prices, times, volumes = load_time_series(select_pair, None, True)
    buy_price_df = pd.DataFrame()
    buy_price_df['times'] = times
    buy_price_df["price_buy"] = prices
    buy_price_df["volume_buy"] = volumes
    buy_price_df.set_index('times', inplace=True)
    buy_price_df.fillna(method='ffill', inplace=True)

    prices, times, volumes = load_time_series(select_pair, None, False)
    sell_price_df = pd.DataFrame()
    sell_price_df['times'] = times
    sell_price_df["price_sell"] = prices
    sell_price_df["volume_sell"] = volumes
    sell_price_df.set_index('times', inplace=True)
    sell_price_df.fillna(method='ffill', inplace=True)

    price_df = buy_price_df.join(sell_price_df)
    price_df["prices"] = price_df.apply(lambda x: (x["price_buy"] + x["price_sell"]) * 0.5, axis=1)
    price_df.reset_index(inplace=True)

    '''
    cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

    cursor = cnx.cursor()


    
    query = ("SELECT * FROM historic_news_barrier_probs where \
                        currency_pair = '" + select_pair + "' and model_key='" + sys.argv[2] + "' order by time_stamp \
                        ")

    cursor.execute(query)

    print ("out")

    setup_rows = []
    for row1 in cursor:
        setup_rows.append(row1)

        print (len(setup_rows))

    cursor.close()

    print len(setup_rows), "tot"


    pickle.dump(setup_rows, open("rows" + select_pair + ".pickle", "wb"))
    '''
    setup_rows = pickle.load(open("rows" + select_pair + ".pickle", "rb"))
    

    X_train, y_train, times = back_test(select_pair, price_df, setup_rows)
    pickle.dump([X_train, y_train, times], open("signal_training" + select_pair + ".pickle", "wb"))

def process_training_set(select_pair):

    data = pickle.load(open("signal_training" + select_pair + ".pickle", "rb"))
    X_train = data[0]
    y_train = data[1]
    times = data[2]

    print ("loaded")
    y_test_all = []
    y_preds_all = []
    for c_index in range(10):

        X_select = []
        y_select = []

        X_test = []
        y_test = []
        offset = 0
        for x1, y1, t1 in zip(X_train, y_train, times):

            offset += 1

            if (offset % 10) == c_index:
                X_test.append(x1)
                y_test.append(y1)
            else:
                X_select.append(x1)
                y_select.append(y1)


        for y_index in range(19, 20):

            try:
                y_t = [y_l[y_index] for y_l in y_select]
                barrier_clf = xgb.XGBClassifier(seed=1, n_estimators=200, scale_pos_weight = float(sum([y_l == False for y_l in y_t]))/sum(y_t))
                barrier_clf.fit(np.array(X_select), y_t)

                probs = barrier_clf.predict_proba(X_select)[:,1]
                mean_prob = np.mean(probs)

                probs = barrier_clf.predict_proba(X_test)[:,1]

                probs = [p - mean_prob for p in probs]
                label = [y_l[y_index] for y_l in y_test]

                y_test_all += label
                y_preds_all += probs
            except:
                pass

        fpr, tpr, thresholds = metrics.roc_curve(y_test_all, y_preds_all)

        print("auc", metrics.auc(fpr, tpr))


create_training_set("EUR_JPY")
process_training_set("EUR_JPY")


