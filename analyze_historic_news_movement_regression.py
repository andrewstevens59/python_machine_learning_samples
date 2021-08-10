import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from bayes_opt import BayesianOptimization
from datetime import timedelta
import execute_news_signals
from execute_news_signals import ModelType
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

            if high != low or utc.weekday() in {4}:
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


def barrier_function(prev_releases, avg_probability_low_barrier, currency_pair, is_norm_prob, is_norm_base, auc_barrier_mult, is_low_barrier, max_barrier, currency_weights, max_release_time_delay):

    avg_dir = 0
    avg_count = 0
    aucs = []
    probs = []

    found = False

    time_delay_map = {}
    for release in prev_releases:
        time_stamp = release[2]
        relase_time = release[3]
        currency = release[0]

        barrier = release[4]

        '''
        if is_low_barrier:
            prob = release[5] - avg_probability_low_barrier[barrier]
        else:
            key = str(barrier) + "_" + str(auc_barrier_mult)
            if key in avg_probability_high_barrier:
                prob = release[5] - avg_probability_high_barrier[key]
            else:
                prob = release[5]
        '''

        if currency_weights != None and currency in currency_weights:
            currency_weight = currency_weights[currency]
        else:
            currency_weight = 1.0

        if currency_weight < 0.01:
            continue

        if barrier in avg_probability_low_barrier:
            prob = release[5] - avg_probability_low_barrier[barrier]
        else:
            prob = release[5]

        auc = release[6]

        '''
        key = str(time_stamp) + "_" + str(relase_time)
        if key not in time_delay_map:
            hours = calculate_time_diff(time_stamp, relase_time)
            time_delay_map[key] = hours
        else:
            hours = time_delay_map[key]


        if hours > max_release_time_delay:
            continue
        '''

        if (currency != currency_pair[0:3] and currency != currency_pair[4:7] and is_relavent_currency == True):
            continue

        if barrier > max_barrier:
            continue

        if abs(prob - 0.5) < 0.5 - max(0, (auc - 0.5) * auc_barrier_mult) and is_low_barrier == False:
            continue

        if is_norm_prob:
            if prob > 0.5:
                prob = 1.0
            else:
                prob = 0.0

        if auc > 0.51:
            avg_dir += barrier * (prob - 0.5) * currency_weight

            if is_norm_base:
                avg_count += abs(prob - 0.5)  * currency_weight
            else:
                avg_count += (prob - 0.5) * currency_weight

            found = True

        exchange_rate = release[9]

    if found:

        for release in prev_releases:
            auc = release[6]
            prob = release[5]
            aucs.append(0.5)

    return avg_dir, avg_count, aucs, probs

def time_decay_function_regression(prev_releases, avg_probability_low_barrier, currency_pair, is_norm_prob, is_norm_base, auc_barrier_mult, is_low_barrier, max_barrier, currency_weights, max_release_time_delay):

    avg_dir = 0
    avg_count = 0
    aucs = []
    probs = []

    found = False

    for release in prev_releases:
        time_stamp = release[2]
        relase_time = release[3]
        currency = release[0]

        barrier = release[4]

        prob = release[5]

        auc = release[6]

        if (currency != currency_pair[0:3] and currency != currency_pair[4:7] and is_relavent_currency == True):
            continue

        if barrier > max_barrier:
            continue
        
        
        if abs(prob) * auc_barrier_mult < auc and is_low_barrier == False:
            continue

        avg_dir += prob
        avg_count += 1

        exchange_rate = release[9]

    if found:

        for release in prev_releases:
            auc = release[6]
            prob = release[5]
            aucs.append(0.5)

    return avg_dir, avg_count, aucs, probs

def time_decay_function_binary(prev_releases, avg_probability_low_barrier, currency_pair, is_norm_prob, is_norm_base, auc_barrier_mult, is_low_barrier, max_barrier):

    avg_dir = 0
    avg_count = 0
    aucs = []
    probs = []

    found = False

    for release in prev_releases:
        time_stamp = release[2]
        relase_time = release[3]
        currency = release[0]

        barrier = release[4]

        if is_low_barrier:
            prob = release[5] - avg_probability_low_barrier[barrier]
        else:
            key = str(barrier) + "_" + str(auc_barrier_mult)
            if key in avg_probability_high_barrier:
                prob = release[5] - avg_probability_high_barrier[key]
            else:
                prob = release[5]

        auc = release[6]

        if (currency != currency_pair[0:3] and currency != currency_pair[4:7] and is_relavent_currency == True):
            continue

        if barrier > max_barrier:
            continue
        
        
        if abs(prob - 0.5) < 0.5 - max(0, (auc - 0.5) * auc_barrier_mult) and is_low_barrier == False:
            continue

        if is_norm_prob:
            if prob > 0.5:
                prob = 1.0
            else:
                prob = 0.0

        if auc > 0.51:
            avg_dir += 50 * (prob - 0.5)
            avg_count += 1

            found = True

        exchange_rate = release[9]

    if found:

        for release in prev_releases:
            auc = release[6]
            prob = release[5]
            aucs.append(0.5)

    return avg_dir, avg_count, aucs, probs

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
    return (Sxy * N - Sy * Sx)/det, (Sxx * Sy - Sx * Sxy)/det


def find_std(price_df, curr_time, lag):

    before_prices = price_df["prices"][price_df["times"] < curr_time].tail(lag).values.tolist()

    a, b = linreg(range(len(before_prices)),before_prices)

    return a, np.std([before_prices[index] - ((a * index) + b) for index in range(len(before_prices))])


def back_test(select_pair, adjust_factor, is_hedge, auc_barrier_mult, 
    is_norm_base, is_any_barrier, max_barrier, is_norm_prob, 
    is_norm_signal, is_max_volatility, model_type, day_wait, avg_probability_high_barrier,
     max_pip_barrier, reward_risk_ratio, max_order_size, min_trade_volatility,
     predictions, max_pip_slippage, grad_mult, decay_frac):

    orders = []
    curr_trade_dir = None
    ideal_position = 0
    total_profit = 5000
    min_profit = 0
    max_profit = 0

    pnl = 0
    max_exposure = 0
    last_order_time = 0
    equity = []

    equity_buffer = []
    float_pnls = []
    pnls = []
    pip_diffs = []
    trade_time_gaps = []

    min_time = 99999999999999999
    max_time = 0

    max_equity = 0

    barrier_percentiles = {}
    barrier_std = {}
    for barrier_index in predictions[2007]:

        forecasts = []
        for select_year in range(2007, 2010):

            for time_index in range(len(predictions[select_year][barrier_index]["timestamp"])):
                price_forecasts = []
                for price_index in range(0, 5):
                    predict = predictions[select_year][barrier_index]["predictions"][price_index][time_index]
                    price_forecasts.append(predict)

                forecasts.append(np.mean(price_forecasts))

        mean = np.mean(forecasts)
        forecasts = [abs(v) - mean for v in forecasts]

        barrier_percentiles[barrier_index] = np.percentile(forecasts, auc_barrier_mult * 100)
        barrier_std[barrier_index] = np.std(forecasts)



    last_time_stamp = 0
    hit_stop_loss_count = 0
    last_signal_time = 0
    for select_year in range(2007, 2010):

        forecasts = {}
        for barrier_index in predictions[select_year]:

            for time_index in range(len(predictions[select_year][barrier_index]["timestamp"])):
               
                timestamp = predictions[select_year][barrier_index]["timestamp"][time_index]

                price_forecasts = []
                for price_index in range(0, 5):
                    predict = predictions[select_year][barrier_index]["predictions"][price_index][time_index]
                    price_forecasts.append(predict)

                predict = np.mean(price_forecasts)

                if timestamp not in forecasts:
                    forecasts[timestamp] = []

                if abs(predict) > barrier_percentiles[barrier_index]:
                    forecasts[timestamp].append(predict)


        series = [[a, b] for a, b in zip(predictions[select_year][5]["timestamp"], predictions[select_year][5]["prices"])]
        series = sorted(series, key=lambda x: x[0])


        if select_pair[4:7] == "JPY":
            pip_size = 0.01
        else:
            pip_size = 0.0001

        for item in series:

            prev_time_stamp = item[0]
            current_price = item[1]

            min_time = min(min_time, prev_time_stamp)
            max_time = max(max_time, prev_time_stamp)
            time_years = float(max_time - min_time) / (60 * 60 * 24 * 365)

            if is_valid_trading_period(prev_time_stamp) == False:
                continue

            '''
            if abs(prev_time_stamp - last_signal_time) < 60 * 60 * 24 * 7:
                continue
            '''

            if prev_time_stamp not in forecasts:
                continue

            predict = forecasts[prev_time_stamp]
            last_signal_time = prev_time_stamp

            if len(predict) > 0:
                avg_dir = sum(predict)
            else:
                avg_dir = 0

            avg_count = len(predict)
            if avg_count == 0:
                continue

            growth_factor = 1.0


            between_df = price_df[(price_df["times"] >= last_time_stamp) & (price_df["times"] <= prev_time_stamp)]
            between_prices = between_df["prices"].values.tolist()
            between_times = between_df["times"].values.tolist()
            last_time_stamp = prev_time_stamp

            
            # go over prices before release
            if len(between_prices) > 0:
                prev_price = between_prices[0]
            
            for between_price, between_time in zip(between_prices, between_times):

                if len(orders) == 0:
                    break

                pnl = 0
                new_orders = []
                for order in orders:

                    if between_time < order.open_time:
                        continue

                    if order.dir == (between_price > order.open_price):
                        profit = (abs(order.open_price - between_price) - (pip_size * 5)) * order.amount
                    else:
                        profit = (-abs(order.open_price - between_price) - (pip_size * 5)) * order.amount


                    pip_diff = (profit / pip_size) / order.amount
                    if select_pair[4:7] == "JPY":
                        profit /= 100

                    order.pnl = profit

                    if  (((pip_diff < -max_pip_barrier) or (pip_diff > max_pip_barrier * reward_risk_ratio))):
                        total_profit += profit

                        limit_profit = max(-max_pip_barrier * order.amount, profit)
                        limist_profit = min(max_pip_barrier * order.amount * reward_risk_ratio, profit)

                        limit_profit = max(-50 * growth_factor * adjust_factor * 2, profit)
                        limit_profit = min(50 * reward_risk_ratio * growth_factor * adjust_factor * 2, profit)
                        pnls.append(limit_profit * (1.0 / order.growth_factor))

                        if profit < 0:
                            hit_stop_loss_count += 1
                        continue

                    pnl += profit
                    new_orders.append(order)

                orders = new_orders
            
            
            grad, std1 = find_std(price_df, prev_time_stamp, 24 * 20) 
            grad /= pip_size
            std1 /= pip_size

            pnl = 0
            total_buy = 0
            total_sell = 0
            order_count = 0
            total_amount = 0
            for order in orders:

                if order.dir == (current_price > order.open_price):
                    profit = (abs(order.open_price - current_price) - (pip_size * 5)) * order.amount
                else:
                    profit = (-abs(order.open_price - current_price) - (pip_size * 5)) * order.amount

                pip_diff = (profit / pip_size) / order.amount
                if select_pair[4:7] == "JPY":
                    profit /= 100

                pip_diffs.append(abs(pip_diff))
                order.max_pnl = max(order.max_pnl, profit)
                order.pnl = profit
                total_amount += order.amount

                if (order.trade_dir):
                    total_buy += order.amount
                else:
                    total_sell += order.amount
   
                pnl += profit
  

            if len(orders) > 0:
                float_pnls.append(pnl)
                
                if len(float_pnls) > 1000:
                    float_pnls = float_pnls[1:]

            if abs(avg_count) > 0:# and (len(orders) == 0 or abs(exchange_rate - price_deltas[-1]) > pip_size * 10):
                equity.append(total_profit + pnl)

                if is_norm_signal:
                    signal = avg_dir / abs(avg_count)
                else:
                    signal = avg_dir

                signal = min(signal, 200)
                signal = max(signal, -200)

                trade_time_gaps.append(prev_time_stamp)

                if abs(signal) > 0 and std1 > min_trade_volatility:

                    max_equity = max(max_equity, total_profit + pnl)

                    if signal > 0:
                        amount = abs(signal) * ((900000 * 0.0001 * growth_factor) + (0)) * 0.5
                    else:
                        amount = -abs(signal) * ((900000 * 0.0001 * growth_factor) + (0)) * 0.5
                
                    if pnl > abs(np.mean(float_pnls)) * 1.2:
                        for order in orders:
                            pnls.append(order.pnl * (1.0 / order.growth_factor))
                        total_profit += pnl
                        orders = []
                        pnl = 0
                        curr_trade_dir = total_buy < total_sell

                    if len(orders) > 0:
                        ideal_position = (ideal_position * (1 - decay_frac)) + (amount * decay_frac)
                        delta_pos = ideal_position - (total_buy - total_sell)
                        delta_fraction = abs(delta_pos) / abs(total_buy - total_sell)
                    else:
                        ideal_position = amount
                        delta_pos = amount
                        delta_fraction = 1.0

                    amount = abs(delta_pos)
                    signal = delta_pos
                    total_amount += amount

                    if (delta_fraction > 0.1) and total_amount < (total_profit + pnl) * 50:

                        if is_hedge == False:
                            temp_orders = orders
                            new_orders = []
                        else:
                            temp_orders = []
                            new_orders = orders

                        for curr_order in temp_orders:
                            if (amount < 0) or ((signal > 0) == curr_order.trade_dir):
                                new_orders.append(curr_order)
                                continue
                                
                            if amount >= curr_order.amount:
                                total_profit += curr_order.pnl
                                amount -= curr_order.amount
                                pnls.append(curr_order.pnl * (1.0 / order.growth_factor))

                            else:
                                total_profit += curr_order.pnl * (amount / curr_order.amount)
                                pnls.append(curr_order.pnl * (amount / curr_order.amount) * (1.0 / order.growth_factor))
                                curr_order.amount -= amount
                                new_orders.append(curr_order)
                                amount = -1

                        orders = new_orders

                        if amount > 0:
                            order = Order()
                            order.open_price = current_price
                            order.trade_dir = signal > 0 
                            order.amount = amount
                            order.open_time = prev_time_stamp
                            order.growth_factor = growth_factor
                            curr_trade_dir = (signal > 0)

                            if order.trade_dir:
                                total_buy += amount
                            else:
                                total_sell += amount

                            orders.append(order)

            max_exposure = max(max_exposure, total_amount / ((total_profit + pnl) * 50))

            max_profit = max(max_profit, total_profit + pnl)
            min_profit = min(min_profit, total_profit + pnl - max_profit)


            equity_buffer.append(total_profit + pnl)


    print "Sharpe", np.mean(pnls) / np.std(pnls)
    print "Samples", len(pnls) / time_years
    print "time_between_trades", np.mean([(a - b) / (60 * 60 * 24) for a, b in zip(trade_time_gaps[1:], trade_time_gaps[:-1])])

    if abs(time_years) > 0:
        print total_profit + pnl, ((((total_profit + pnl - 5000) / 5000)) / time_years), time_years, len(orders), min_profit, max_exposure

    if len(pnls) == 0:
        return -100, [0], 0, 0, 0

    return np.mean(pnls) / np.std(pnls), equity, float(hit_stop_loss_count) / len(pnls), np.mean(pnls), len(pnls) / time_years


def search3(setting):

    if "min_trade_volatility" not in setting:
        setting["min_trade_volatility"] = 50

    if "decay_frac" not in setting:
        setting["decay_frac"] = 0.5

    pbounds = {
            'adjust_factor': (setting["adjust_factor"], setting["adjust_factor"]),
            'day_wait' : (500000, 500000),
            'auc_barrier' : (0.0, 1.0),
            'is_norm_signal' : (0, 0),
            'max_barrier' : (100, 100),
            'is_max_volatility' : (setting["is_max_volatility"], setting["is_max_volatility"]),
            'max_pip_barrier' : (setting["max_pip_barrier"], setting["max_pip_barrier"]),
            'reward_risk_ratio' : (setting["reward_risk_ratio"], setting["reward_risk_ratio"]),
            'max_order_size' : (200, 200),
            'min_trade_volatility' : (20, 150),
            'max_pip_slippage' : (3.0, 3.0),
            'grad_mult' : (1, 1),
            'decay_frac' : (0.1, 0.1)
            }

    return pbounds


def bayesian_optimization_output(setting, select_pair, predictions, is_hedge,
    is_any_barrier, model_type):

    pbounds = search3(setting)
    #pbounds = search1(setting, is_relavent_currency, select_pair)
    
    all_sharpes = []
    samples_set = []

    def xgboost_hyper_param(adjust_factor, day_wait, auc_barrier, is_norm_signal, 
        max_barrier, is_max_volatility, max_pip_barrier, reward_risk_ratio, max_order_size,
        min_trade_volatility, max_pip_slippage, grad_mult, decay_frac):

        if "currency_weights" in setting:
            currency_weights = setting["currency_weights"]
        else:
            currency_weights = None

        sharpe, equity_curve, stop_loss_ratio, mean_pnl, samples = back_test(select_pair, adjust_factor, is_hedge,
                                            auc_barrier, True, False, 
                                            max_barrier, False, is_norm_signal > 0.5, 
                                            is_max_volatility > 0.5, model_type, day_wait,
                                             {}, max_pip_barrier, reward_risk_ratio, max_order_size,
                                            min_trade_volatility, predictions, max_pip_slippage, 
                                            grad_mult, decay_frac)

        if samples < 10:
            return -1

        all_sharpes.append(sharpe)
        samples_set.append(samples)
        
        return sharpe * max(1, (equity_curve[-1] / 5000))
     
     
    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
    )

    optimizer.maximize(
        init_points=4,
        n_iter=8,
    )

    max_sharpe = max(all_sharpes)
    samples = [sample for sharpe, sample in zip(all_sharpes, samples_set) if sharpe >= max_sharpe]

    return optimizer.max['params'], optimizer.max['target'], max_sharpe, samples[0], np.mean(all_sharpes)



model_type = sys.argv[3]
if model_type == "barrier":
    model_type = ModelType.barrier
elif model_type == "time_regression":
    model_type = ModelType.time_regression
elif model_type == "time_classification":
    model_type = ModelType.time_classification

def rank_data_size():
    final_fitted_map = execute_news_signals.get_strategy_parameters(model_type)
   #final_fitted_map = pickle.load(open(str(model_type) + "_final_fitted_map.pickle", "rb"))
    
    for j in range(4):
        ranking = {}
        for i in [j]:
            s3 = []
            for pair in final_fitted_map:
                if pair not in all_currency_pairs:
                    continue


                if pair not in ranking:
                    ranking[pair] = []

                ranking[pair].append(final_fitted_map[pair][i]['sharpe'] / final_fitted_map[pair][i]['samples'])

    final_ranking = []
    for pair in final_fitted_map:
        if pair not in all_currency_pairs:
             continue

        if final_fitted_map[pair+"_sample_num"] > 1202353:
            continue

        ranking[pair] = np.mean(ranking[pair])

        final_ranking.append([ranking[pair], pair])

    final_ranking = sorted(final_ranking, key=lambda x: x[0], reverse=True)
    print ([item[1] for item in final_ranking])






#final_fitted_map = execute_news_signals.get_strategy_parameters(model_type)
final_fitted_map = pickle.load(open(str(model_type) + "_final_fitted_map_currency.pickle", "rb"))


all_sharpes = []

for select_pair in all_currency_pairs:

    print (select_pair)

    fp=gzip.open("models/historic_predictions_" + select_pair + ".gz", "rb")
    predictions = cPickle.load(fp)
    fp.close()


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


    settings = []

    aucs = execute_news_signals.get_strategy_parameters(ModelType.barrier)[select_pair]
    
    setting_offset = 0

    for is_hedge in [False]:

        print ("optimize original ", aucs[2]["target"])
        setting = aucs[len(settings)]
        if "currency_weights" in setting:
            currency_weights = setting["currency_weights"]
        else:
            currency_weights = None

        params, target, sharpe, samples, mean_sharpe = bayesian_optimization_output(setting, select_pair, predictions, 
            is_hedge, False, model_type)

        if True or target > setting["target"] or (setting["samples"] < 10 and target > 0):
            all_sharpes.append(target)
            print ("Found Better")

            new_setting = {
                "is_low_barrier" : False,
                "is_any_barrier" : False,
                "is_hedge" : is_hedge,
                "decay_frac" : params['decay_frac'],
                "adjust_factor" : params['adjust_factor'],
                "auc_barrier" : params['auc_barrier'],
                "is_norm_signal" : params['is_norm_signal'] > 0.5,
                "is_max_volatility" : params['is_max_volatility'] > 0.5,
                "currency_pair" : select_pair,
                "max_pip_barrier" : params['max_pip_barrier'],
                "reward_risk_ratio" : params['reward_risk_ratio'],
                "max_order_size" : params['max_order_size'],
                "min_trade_volatility" : params['min_trade_volatility'],
                "samples" : samples,
                "sharpe" : sharpe,
                "target" : target,
                "mean_sharpe" : mean_sharpe,
                }

            currency_weights = {}
            for currency in ['AUD', 'GBP', 'CAD', 'EUR', 'NZD', 'CHF', 'USD', 'JPY']:
                if currency in params:
                    currency_weights[currency] = params[currency]

            if len(currency_weights) == 0 and "currency_weights" in setting:
                currency_weights = setting["currency_weights"]

            new_setting['currency_weights'] = currency_weights
            settings.append(new_setting)
        else:
            all_sharpes.append(setting["sharpe"])
            settings.append(setting)


        print ("Mean Sharpe Overall", np.mean(all_sharpes))


    final_fitted_map[select_pair] = settings
    pickle.dump(final_fitted_map, open(str(model_type) + "_final_fitted_map_currency.pickle", "wb"))

    cursor.close()

    '''
    plt.title("Best Adjust: " + str(best_adjust_factor))
    plt.plot(best_equity_curve)
    plt.show()
    '''


default_stops = {False : [], True : []}
default_auc_mults = []
default_norm_signals = []
default_wait_days = {False : [], True : []}

for pair in final_fitted_map:

    if pair not in all_currency_pairs:
        continue

    print (pair)

    for setting in final_fitted_map[pair]:
        default_auc_mults.append(setting["auc_barrier"])
        default_norm_signals.append(setting["is_norm_signal"])
        default_stops[setting["is_hedge"]].append(setting["adjust_factor"])
        default_wait_days[setting["is_hedge"]].append(setting["day_wait"])

print (len(default_norm_signals))

final_fitted_map["default_day_wait"] = {False : np.mean(default_wait_days[False]), True : np.mean(default_wait_days[True])}
final_fitted_map["default_adjust"] = {False : np.mean(default_stops[False]), True : np.mean(default_stops[True])}
final_fitted_map["default_auc_mult"] = np.mean(default_auc_mults)
final_fitted_map["default_is_norm_signal"] = sum(default_norm_signals) > sum([(v == False) for v in default_norm_signals])

print (final_fitted_map)

print ("Default Day Wait: ", final_fitted_map["default_day_wait"])
print ("Default Adjust: ", final_fitted_map["default_adjust"])
print ("Default AUC Mult: ", final_fitted_map["default_auc_mult"])
print ("Default default_is_norm_signal: ", final_fitted_map["default_is_norm_signal"])


'''
('Mean Sharpe Overall', 4748.545318017648)
('Mean Return Overall', 4748.545318017648)
'''



