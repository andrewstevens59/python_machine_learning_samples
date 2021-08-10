import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import json
import copy

import socket
import sys
import time

import math
import sys
import re

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
import download_calendar as download_calendar
import gzip, cPickle
import string
import random as rand

from os import listdir
from os.path import isfile, join

import os
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from maximize_sharpe import *

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from evaluate_model import evaluate
from uuid import getnode as get_mac
import socket
import paramiko
import json

import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import logging
from close_trade import *
import enum
import os

class ModelType(enum.Enum): 
    barrier = 1
    time_regression = 2
    time_classification = 3

class Order:

    def __init__(self):
        self.pair = ""
        self.dir = 0
        self.open_price = 0
        self.open_time = 0
        self.amount = 0
        self.PnL = 0
        self.tp_price = 0
        self.sl_price = 0
        self.actual_amount = 0
        self.account_number = None
        self.time_diff_hours = 0
        self.order_id = 0
        self.base_amount = 0
        self.sequence_id = 0
        self.model_key = None
        self.prediction_key = None
        self.margin_used = 0
        self.open_prediction = 0
        self.curr_prediction = 0
        self.portfolio_wt = 0
        self.calendar_time = 0
        self.barrier_size = 0
        self.carry_cost = 0
        self.ideal_price = 0
        
currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

import datetime as dt

api_key = None
file_ext_key = ""
account_type = None

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

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_trade_bias(pair, is_buy):

    return 1.0

    df = pd.read_csv('/root/sentiment.csv')
    df = df[df['Symbol'] == pair]

    if len(df) == 0:
        return 1.0

    try:

        if is_buy:
            if float(df['long_perc']) > 50:
                return float(100 - min(50, abs(50 - float(df['long_perc'])))) / 100
            else:
                return float(100 + min(50, abs(50 - float(df['long_perc'])))) / 100
        else:
            if float(df['short_perc']) > 50:
                return float(100 - min(50, abs(50 - float(df['short_perc'])))) / 100
            else:
                return float(100 + min(50, abs(50 - float(df['short_perc'])))) / 100
    except:
        return 1.0

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


formatter = MyFormatter(fmt='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def sendCurlRequest(url, request_type, post_data = None):
    response_buffer = StringIO()
    header_buffer = StringIO()
    curl = pycurl.Curl()

    curl.setopt(curl.URL, url)

    curl.setopt(pycurl.CUSTOMREQUEST, request_type)

    curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

    curl.setopt(curl.HEADERFUNCTION, header_buffer.write)

    #2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8 actual

    #3168adee62ceb8b5f750efcec83c2509-1db7870026de5cccb33b220c100a07ab demo

    #16736a148307bf5d91f9f03dd7c91623-af6a25cf6249c9083f11593ad3899f89 demo2

    print url


    if post_data != None:
        print post_data
        curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer ' + api_key, 'Content-Type: application/json'])
        curl.setopt(pycurl.POSTFIELDS, post_data)
    else:
        curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer ' + api_key])

    curl.perform()
    curl.close()

    response_value = response_buffer.getvalue()
    header_value = header_buffer.getvalue()

    return response_value, header_value

def get_open_trades(account_number, order_metadata, total_margin, select_pair):

    orders = []
    pair_time_diff = {}
    next_link = "https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades?count=50&instrument=" + select_pair

    while next_link != None:
        response_value, header_value = sendCurlRequest(next_link, "GET")

        lines = header_value.split('\n')
        lines = filter(lambda line: line.startswith('Link:'), lines)

        if len(lines) > 0:
            line = lines[0]
            next_link = line[line.find("<") + 1 : line.find(">")]
        else:
            next_link = None

        j = json.loads(response_value)
        open_orders = j['trades']
        open_orders = sorted(open_orders, key=lambda order: order['openTime'])

        for trade in open_orders:
            s = trade['openTime'].replace(':', "-")
            s = s[0 : s.index('.')]
            order_id = trade['id']
            open_price = float(trade[u'price'])
            pair = trade[u'instrument']
            amount = float(trade[u'currentUnits'])
            pair = trade['instrument'].replace("/", "_")
            PnL = float(trade['unrealizedPL'])
            margin_used = float(trade['marginUsed'])

            if pair != select_pair:
                continue

            pip_size = 0.0001
            if pair[4:7] == "JPY":
                pip_size = 0.01

            time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

            time_diff_hours = calculate_time_diff(time.time(), time_start)

            key = order_id + "_" + account_number
            
            '''
            if key not in order_metadata:
                order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
                trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
                print "not exist", key, order_metadata
                continue
            metadata = order_metadata[key]
                        '''
            order = Order()
            order.open_price = open_price
            order.amount = abs(amount)
            order.pair = pair
            order.dir = amount > 0
            order.time_diff_hours = time_diff_hours
            order.order_id = order_id
            order.account_number = account_number
            order.model_key = select_pair
            order.prediction_key = select_pair
            order.open_time = time_start

            '''
            order.model_key = metadata["model_key"]
            order.open_prediction = metadata["open_prediction"]
            order.curr_prediction = metadata["curr_prediction"]
            order.dir = metadata["dir"]
            order.base_amount = metadata["base_amount"]
            order.sequence_id = metadata["sequence_id"]
            order.prediction_key = metadata["prediction_key"]
            order.calendar_time = metadata["calendar_time"]
            order.barrier_size = metadata["barrier_size"]
            order.ideal_price = metadata["ideal_price"]

            if 'portfolio_wt' in metadata:
                order.portfolio_wt = metadata["portfolio_wt"]
            '''
            order.margin_used = margin_used
            order.PnL = PnL

            orders.append(order)

    return orders, total_margin

def close_order(order):

    order_info, _ =  sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

    if "ORDER_CANCEL" not in order_info:
        trade_logger.info('Close Order: ' + str(order_info)) 
        trade_logger.info("Close Model Order: " + str({'model_key' : order.model_key, "Actual Profit" : str(order.PnL)})) 
        return True

    return False

def create_order(pair, curr_price, account_numbers, trade_dir, order_amount, base_model_key, stop_loss_price, take_profit_price):

    if trade_dir == True or len(account_numbers) == 1:
        account_number = account_numbers[0]
    else:
        account_number = account_numbers[1]


    precision = '%.4f'
    if pair[4:7] == 'JPY':
        precision = '%.2f'

    if trade_dir == True:
        tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % take_profit_price) + '"}'
        sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
        order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ',' + tp_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
    else:
        tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % take_profit_price) + '"}'
        sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
        order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ',' + tp_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
        

    return order_info, account_number, order_amount



def close_group_trades(pair, orders, is_force_close, order_metadata, group_metadata, 
    total_margin_available, total_margin_used, curr_price, 
    avg_prices, avg_spreads, growth_factor, strategy_weight):

    model_key = pair
    stop_loss_weight = growth_factor

    total_pnl = 0
    total_buy = 0
    total_sell = 0
    for order in orders:
        total_pnl += order.PnL
        if order.dir:
            total_buy += order.amount
        else:
            total_sell += order.amount

    trade_logger.info('Model: ' + model_key + \
            ", Total Float Profit: " + str(total_pnl) + \
            ", Net Dir: " + str(total_buy - total_sell) + \
            ", Total Orders: " + str(len(orders)))

    first_currency = pair[0:3]
    second_currency = pair[4:7]

    if second_currency != "AUD":
        pair_mult = avg_prices[second_currency + "_AUD"]
    else:
        pair_mult = 1.0

    now = datetime.datetime.now()

    for time_index in group_metadata["times"]:

        for key in ["buy_orders", "sell_orders"]:
            new_orders = []
            for order in group_metadata["times"][time_index][key]:
                margin_used_factor = order.margin_used / (order.PnL + (total_margin_used))

                time_diff_hours = calculate_time_diff(time.time(), order.open_time)

                trade_logger.info(
                    "    Pair: " + str(order.pair) + \
                    ", Float Profit: " + str(order.PnL) + \
                    ", Amount: " + str(order.amount) + \
                    ", Dir: " + str(order.dir) + \
                    ", Day Wait: " + str(int(time_index / 24)) + \
                    ", Time Diff: " + str('%0.2f' % (time_diff_hours / 24)))

                if time_diff_hours > time_index: 
                    order_metadata, is_success = close_order(order)
                    if is_success == True:
                        continue

                new_orders.append(order)

            group_metadata["times"][time_index][key] = new_orders



def enter_group_trades(pair, orders, growth_factor, prob_dir, order_metadata, group_metadata,
    free_margin, used_margin, curr_price_bid_ask, account_numbers, 
    curr_spread, avg_prices, pair_bid_ask_map, strategy_weight,
    last_signal_update_time, trend_std, time_index, pip_size):

    base_model_key = pair

    if prob_dir > 0:
        curr_price = curr_price_bid_ask['ask']
    else:
        curr_price = curr_price_bid_ask['bid']

    first_currency = pair[0:3]
    second_currency = pair[4:7]

    if second_currency != "AUD":
        pair_mult = avg_prices[second_currency + "_AUD"]
    else:
        pair_mult = 1.0

    if second_currency == "JPY":
        pair_mult *= 100

    update_key = base_model_key + "_last_signal_update_time_" + str(time_index)
    if update_key not in group_metadata:
        group_metadata[update_key] = None

    new_order = Order()
    new_order.dir = (prob_dir > 0)
    order_amount = int(round((abs(prob_dir) * 10000 * growth_factor) / (pair_mult * trend_std)))

    print ("order amount", order_amount, trend_std)

    barrier_size = (trend_std * 2 * pip_size )
    print ("barrier size", barrier_size)

    if order_amount == 0:
        return False

    if group_metadata[base_model_key + "_curr_spread"] >= 2.0:
        print ("spread too big")
        return False

    if group_metadata[update_key] == last_signal_update_time:
        return False


    print growth_factor, pair_mult
    if new_order.dir:
        new_order.ideal_price = pair_bid_ask_map[pair]["ask"]
        stop_loss = new_order.ideal_price - barrier_size
        take_profit = new_order.ideal_price + (barrier_size * 2)
    else:
        new_order.ideal_price = pair_bid_ask_map[pair]["bid"]
        stop_loss = new_order.ideal_price + barrier_size
        take_profit = new_order.ideal_price - (barrier_size * 2)

    order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount, base_model_key, stop_loss, take_profit)

    print str(order_info)

    order_info = json.loads(order_info)

    if 'orderFillTransaction' in order_info:
        trade_logger.info('New Order: ' + str(order_info)) 
        order_id = str(order_info['orderFillTransaction']['id'])
        group_metadata[base_model_key + "_last_profit_udpate"] = time.time()

        last_processed_key = pair + "_last_processed"
        last_price_key = pair + "_last_price"
    
        group_metadata[last_processed_key] = time.time()
        group_metadata[last_price_key] = curr_price
        group_metadata[base_model_key + "_last_order_time"] = time.time()
        group_metadata[base_model_key + "_last_order_dir"] = new_order.dir
        group_metadata[update_key] = last_signal_update_time 

        if time_index not in group_metadata["times"]:
            group_metadata["times"][time_index] = {}
            group_metadata["times"][time_index]["buy_orders"] = []
            group_metadata["times"][time_index]["sell_orders"] = []

        new_order.pair = pair
        new_order.order_id = order_id
        new_order.amount = order_amount
        new_order.account_number = account_number
        new_order.model_key = base_model_key
        new_order.open_time = time.time()

        if new_order.dir:
            group_metadata["times"][time_index]["buy_orders"].append(new_order)
        else:
            group_metadata["times"][time_index]["sell_orders"].append(new_order)

        with open(root_dir + "group_metadata_news_release_" + pair + file_ext_key, "wb") as f:
            pickle.dump(group_metadata, f)
 
    else:
        trade_logger.info('Order Error: ' + str(order_info)) 

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

    index = 0
    while index < len(j):
        item = j[index]

        s = item['time']
        s = s[0 : s.index('.')]
        timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

        times.append(timestamp)
        prices.append(item['closeMid'])
        index += 1

    return prices, times


def process_pending_trades(account_numbers, avg_spreads, select_pair, signals_file_prefix, model_type, is_exponential = False, strategy_weight = 1.0, is_low_barrier = False, is_max_barrier = False, is_new_trade = True):

    total_balance = 0
    total_float_profit = 0
    total_margin_available = 0
    total_margin_used = 0
    for account_number in account_numbers:
        response_value, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/summary", "GET")
        j = json.loads(response_value)

        account_profit = float(j['account'][u'unrealizedPL'])
        account_balance = float(j['account'][u'balance'])
        margin_available = float(j['account']['marginAvailable'])
        margin_used = float(j['account']['marginUsed'])

        total_balance += account_balance
        total_float_profit += account_profit
        total_margin_available += margin_available
        total_margin_used += margin_used

    is_hedge_account = j['account'][u'hedgingEnabled']

    growth_factor = ((total_balance + total_float_profit) / 5000) * strategy_weight

    if os.path.isfile(root_dir + "group_metadata_news_release_" + select_pair + file_ext_key):
        with open(root_dir + "group_metadata_news_release_" + select_pair + file_ext_key, "rb") as f:
            group_metadata = pickle.load(f)
    else:
        group_metadata = {}
        group_metadata["times"] = {}

    order_metadata = {}

    if "max_equity" not in group_metadata:
        group_metadata["max_equity"] = total_balance

    group_metadata["max_equity"] = max(group_metadata["max_equity"], (total_balance + total_float_profit))

    orders = []
    total_margin = 0
    for account_number in account_numbers:
        orders1, total_margin = get_open_trades(account_number, order_metadata, total_margin, select_pair)
        orders += orders1

    if len(orders) == 0 and is_new_trade == False:
        return

    found_orders = set()
    for time_index in group_metadata["times"]:

        for key in ["buy_orders", "sell_orders"]:
            new_orders = []
            for compare_order in group_metadata["times"][time_index][key]:

                found = False
                for order in orders:
                    if order.order_id == compare_order.order_id:
                        found_orders.add(order.order_id)
                        found = True
                        break

                if found == True:
                    new_orders.append(compare_order)

            group_metadata["times"][time_index][key] = new_orders

    for order in orders:
        if order.order_id not in found_orders:
            print ("close order not found", order.order_id)
            close_order(order)

                    

    trade_logger.info('Equity: ' + str(total_balance + total_float_profit))

    total_orders = len(orders)

    avg_prices = {}
    pair_bid_ask_map = {}
    for pair in currency_pairs:

        first_currency = pair[0:3]
        second_currency = pair[4:7]

        if pair != select_pair:
            if first_currency != "AUD" and second_currency != "AUD":
                continue

            if second_currency != select_pair[4:7] and first_currency != select_pair[4:7]:
                continue

        response, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v1/prices?instruments=" + pair, "GET")
        response = json.loads(response)['prices']

        pip_size = 0.0001
        if pair[4:] == "JPY":
            pip_size = 0.01

        bid = None
        ask = None
        for spread_count in range(1):
            curr_spread = 0
            for price in response:
                if price['instrument'] == pair:
                    curr_price = (price['bid'] + price['ask']) / 2
                    curr_spread = abs(price['bid'] - price['ask']) / pip_size
                    bid = price['bid']
                    ask = price['ask']
                    break

            if curr_price == 0:
                print "price not found"
                continue

        pair_bid_ask_map[pair] = {}
        pair_bid_ask_map[pair]['bid'] = bid
        pair_bid_ask_map[pair]['ask'] = ask

        avg_prices[first_currency + "_" + second_currency] = curr_price

    for pair in currency_pairs:

        first_currency = pair[0:3]
        second_currency = pair[4:7]

        if pair != select_pair:
            if first_currency != "AUD" and second_currency != "AUD":
                continue

            if second_currency != select_pair[4:7] and first_currency != select_pair[4:7]:
                continue

        avg_prices[second_currency + "_" + first_currency] = 1.0 / avg_prices[pair]

    last_signal_update_time = time.ctime(os.path.getmtime(signals_file_prefix + select_pair + ".pickle"))
    print ("Last Modified", last_signal_update_time)
    with open(signals_file_prefix + select_pair + ".pickle", "rb") as f:
        base_calendar = pickle.load(f)

    ideal_spread = avg_spreads[select_pair]
    pip_size = 0.0001
    if select_pair[4:] == "JPY":
        pip_size = 0.01

    actual_spread = abs(pair_bid_ask_map[select_pair]['bid'] - pair_bid_ask_map[select_pair]['ask']) / pip_size
    print ("Actual Spread", actual_spread, pair_bid_ask_map[select_pair]['bid'], pair_bid_ask_map[select_pair]['ask'])
    actual_spread1 = actual_spread
    if actual_spread > 4:
        actual_spread /= ideal_spread 
    else:
        actual_spread = 0

    if select_pair + "_prev_dir" not in group_metadata:
        curr_price = abs(pair_bid_ask_map[select_pair]['bid'] + pair_bid_ask_map[select_pair]['ask']) / 2
    else:
        if group_metadata[select_pair + "_prev_dir"] == True:
            curr_price = pair_bid_ask_map[select_pair]['bid']
        else:
            curr_price = pair_bid_ask_map[select_pair]['ask']

    curr_spread = actual_spread

    last_processed_key = select_pair + "_last_processed"
    last_price_key = select_pair + "_last_price"
    if last_price_key not in group_metadata:
        group_metadata[last_price_key] = curr_price

    price_diff = abs(curr_price - group_metadata[last_price_key]) / pip_size

    print select_pair, "Curr Spread", curr_spread, price_diff

    print ("pip diff", price_diff, ideal_spread, curr_price, group_metadata[last_price_key], pip_size)
    
    group_metadata[select_pair + "_curr_spread"] = curr_spread

    if curr_spread >= 2.0:
        trade_logger.info("Spread too big " + str(actual_spread1) + " " + str(ideal_spread))

    orders = close_group_trades(select_pair, orders, False, \
        order_metadata, group_metadata, total_margin_available, total_margin_used, \
        curr_price, avg_prices, avg_spreads, \
        growth_factor, strategy_weight)

    time_forecast = {}
    time_std = {}
    if select_pair in base_calendar:

        for prediction in base_calendar[select_pair]:

            net_sharpe = 0
            for compare in base_calendar[select_pair]:
                if compare["samples"] > 20 and compare["time_wait"] >= prediction["time_wait"]:
                    net_sharpe += compare["sharpe"]


            if net_sharpe < 0 or prediction["sharpe"] < 0:
                continue

            if prediction["samples"] < 30:
                continue

            time_index = prediction["time_wait"]

            if time_index not in time_forecast:
                time_forecast[time_index] = []
                time_std[time_index] = prediction["std_actuals"]

            if prediction["sharpe"] > 0.07:
                if prediction["forecast"] > 0:
                    time_forecast[time_index].append(abs(prediction["sharpe"]))
                else:
                    time_forecast[time_index].append(-abs(prediction["sharpe"]))

    
    for time_index in time_forecast:
        signal = sum(time_forecast[time_index]) 

        print (time_index, signal)

        if time_index in group_metadata["times"]:

            if len(group_metadata["times"][time_index]["buy_orders"]) != len(group_metadata["times"][time_index]["sell_orders"]):
                if (signal > 0) == (len(group_metadata["times"][time_index]["buy_orders"]) > len(group_metadata["times"][time_index]["sell_orders"])):
                    continue

        trade_logger.info('Signal: ' + str(time_index) + " " + str(signal))
 
        base_model_key = enter_group_trades(select_pair, orders, growth_factor, signal, order_metadata, \
            group_metadata, total_margin_available, total_margin_used, pair_bid_ask_map[select_pair], \
            account_numbers, curr_spread, avg_prices, pair_bid_ask_map, strategy_weight,
            last_signal_update_time, time_std[time_index], time_index, pip_size)

    with open(root_dir + "group_metadata_news_release_" + select_pair + file_ext_key, "wb") as f:
        pickle.dump(group_metadata, f)
    #pickle.dump(order_metadata, open(root_dir + "order_metadata_news_release_" + select_pair + file_ext_key, 'wb'))

    trade_logger.info('Float Profit: ' + str(total_float_profit)) 
    trade_logger.info('Total Orders: ' + str(total_orders)) 
    trade_logger.info('Total Margin: ' + str(total_margin_used)) 
    trade_logger.info('Max Value: ' + str(group_metadata["max_equity"])) 


accounts = [
    ["101-011-9454699-002", "101-011-9454699-003"],
    ["101-011-9454699-004", "101-011-9454699-005"],
    ["101-011-9454699-006", "101-011-9454699-007"]
]

if get_mac() != 150538578859218:
    avg_spreads = pickle.load(open("/root/pair_avg_spread", 'rb'))
    root_dir = "/root/" 
else:
    avg_spreads = pickle.load(open("pair_avg_spread", 'rb'))
    root_dir = "/tmp/" 


import psutil

def checkIfProcessRunning(processName, command):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 3 and processName.lower() in cmdline[2] and command == cmdline[3]:
                count += 1
            elif len(cmdline) > 2 and processName.lower() in cmdline[1] and command == cmdline[2]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 3:
        sys.exit(0)




