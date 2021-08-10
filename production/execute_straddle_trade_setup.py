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
import gzip, cPickle
import string
import random as rand

from os import listdir
from os.path import isfile, join
from lxml.html import fromstring

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
import logging
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

api_key = "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"
file_ext_key = ""
account_type = "fxpractice"

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

def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            #Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies

def sendCurlRequest(url, request_type, post_data = None):

    #print (get_proxies())
    response_buffer = StringIO()
    header_buffer = StringIO()
    curl = pycurl.Curl()
    #curl.setopt(pycurl.PROXY,"http://190.214.9.34:4153") 

    curl.setopt(curl.URL, url)

    curl.setopt(pycurl.CUSTOMREQUEST, request_type)

    curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

    curl.setopt(curl.HEADERFUNCTION, header_buffer.write)


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

def get_open_trades(account_number, order_metadata, total_margin, select_pair, is_force_close = False):

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
            
            
            if is_force_close:
                order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
                trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
                continue
                        
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

def close_order(select_pair, order, group_metadata, existing_order_amount, reduce_amount = None):

    if reduce_amount != None:
        metadata = '{"units": "' + str(reduce_amount) + '"}'
        order_info, _ =  sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT", metadata)
    else:
        order_info, _ =  sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

    if "ORDER_CANCEL" in order_info:
        return group_metadata, existing_order_amount, False


    if group_metadata != None: 
        if "close_pnls" not in group_metadata:
            group_metadata["close_pnls"] = []
            group_metadata["close_pnls_times"] = []

        group_metadata["close_pnls"].append(order.PnL / group_metadata[select_pair + "_growth_factor"])
        group_metadata["close_pnls_times"].append(time.time())
        while len(group_metadata["close_pnls"]) > 1 and time.time() - group_metadata["close_pnls_times"][0] > 60 * 60 * 24 * 172:
            group_metadata["close_pnls"] = group_metadata["close_pnls"][1:]
            group_metadata["close_pnls_times"] = group_metadata["close_pnls_times"][1:]

        curr_profit = 0
        max_profit = 0
        for profit in group_metadata["close_pnls"]:
            curr_profit += profit
            max_profit = max(max_profit, profit)

        group_metadata["recover_factor"] = 1 + (abs(group_metadata["close_pnls"][-1] - max_profit) * 0.05)
        trade_logger.info('Recover Factor: ' + str(group_metadata["recover_factor"])) 

    trade_logger.info('Close Order: ' + str(order_info)) 
    trade_logger.info("Close Model Order: " + str({'model_key' : order.model_key, "Actual Profit" : str(order.PnL)})) 
    return group_metadata, existing_order_amount, True

 


def adjust_orders(pair, orders, delta_amount, trade_dir, group_metadata):

    for order in orders:

        if order.dir != trade_dir:
            if abs(delta_amount) > order.amount:
                if close_order(pair, order, group_metadata, None):
                    delta_amount -= order.amount
                    continue
            else:
                if close_order(pair, order, group_metadata, None, reduce_amount = delta_amount):
                    delta_amount = 0
                    break

    return delta_amount


def create_order(pair, curr_price, account_numbers, trade_dir, order_amount, existing_order_amount, base_model_key, stop_loss_price, take_profit_price):

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

def update_orders(orders, group_metadata, order_metadata):

    order_metadata = {}

    sequence_id = 0
    for order in orders:

        order.sequence_id = sequence_id
        sequence_id += 1

        order_metadata[str(order.order_id) + "_" + order.account_number] = {
            "model_key" : order.model_key,
            "base_amount" : order.base_amount,
            "sequence_id" : order.sequence_id,
            "account_number" : order.account_number,
            "open_prediction" : order.open_prediction,
            "curr_prediction" : order.curr_prediction,
            "prediction_key" : order.prediction_key,
            "portfolio_wt" : order.portfolio_wt,
            "calendar_time" : order.calendar_time,
            "amount" : order.amount,
            "barrier_size" : order.barrier_size,
            "ideal_price" : order.ideal_price,
            "dir" : order.dir,
        }
    return order_metadata

def serialize_order(order):

    return {
            "model_key" : order.model_key,
            "base_amount" : order.base_amount,
            "sequence_id" : order.sequence_id,
            "account_number" : order.account_number,
            "open_prediction" : order.open_prediction,
            "curr_prediction" : order.curr_prediction,
            "prediction_key" : order.prediction_key,
            "portfolio_wt" : order.portfolio_wt,
            "calendar_time" : order.calendar_time,
            "amount" : order.amount,
            "barrier_size" : order.barrier_size,
            "ideal_price" : order.ideal_price,
            "dir" : order.dir,
        }



def close_group_trades(pair, orders, is_force_close, order_metadata, group_metadata, 
    total_margin_available, total_margin_used, existing_order_amount, growth_factor, 
    is_exponential, adjust_factor, strategy_weight, reward_risk_ratio, 
    is_exceed_trade_volatility, is_show_log = True):

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

    if model_key + "_min_profit" not in group_metadata:
        group_metadata[model_key + "_min_profit"] = total_pnl

    group_metadata[model_key + "_min_profit"] = min(total_pnl, group_metadata[model_key + "_min_profit"])

    if is_show_log:
        trade_logger.info('Model: ' + model_key + \
                ", Total Float Profit: " + str(total_pnl / growth_factor) + \
                ", Total Orders: " + str(len(orders)))

    if model_key + "_reduce_amount" not in group_metadata:
        group_metadata[model_key + "_reduce_amount"] = len(orders)

    now = datetime.datetime.now()

    if len(orders) > 0:
        min_order_time = min([order.open_time for order in orders])
        time_diff = calculate_time_diff(time.time(), min_order_time)

    new_orders = []
    for order in orders:
        margin_used_factor = order.margin_used / (order.PnL + (total_margin_used))
        time_diff_now = calculate_time_diff(time.time(), order.open_time) / 24

        if is_show_log:
            trade_logger.info(
                "    Pair: " + str(order.pair) + \
                ", Float Profit: " + str(order.PnL) + \
                ", Amount: " + str(order.amount) + \
                ", Dir: " + str(order.dir) + \
                ", Max Loss: " + str(total_pnl  < -50 * stop_loss_weight * adjust_factor) + \
                ", Time Diff: " + str('%0.2f' % time_diff_now))

        is_time_elapse = (is_exceed_trade_volatility == False) and time_diff > 20

        if is_time_elapse or is_force_close or ((total_pnl  < -50 * stop_loss_weight * adjust_factor) or (total_pnl  > 50 * stop_loss_weight * adjust_factor * reward_risk_ratio)): 
            order_metadata, existing_order_amount, is_success = close_order(pair, order, group_metadata, existing_order_amount)
            if is_success == True:
                continue

        new_orders.append(order)

    if len(new_orders) > 0:
        group_metadata[model_key + "_buy_sell"] = total_buy > total_sell
        group_metadata[model_key + "_pnl"] = total_pnl

    if len(new_orders) == 0 and (model_key + "_buy_sell") in group_metadata:
        if group_metadata[model_key + "_pnl"] > 0 or total_pnl > 0:
            group_metadata[model_key + "_last_dir"] = None  #(group_metadata[model_key + "_buy_sell"] == False)
        else:
            group_metadata[model_key + "_last_dir"] = group_metadata[model_key + "_buy_sell"]

        del group_metadata[model_key + "_buy_sell"]

    if len(new_orders) != len(orders):
        order_metadata = update_orders(new_orders, group_metadata, order_metadata)

    if is_force_close and len(orders) > 0:
        time.sleep(2)

    return new_orders

def check_enter_trade(pair, orders, growth_factor, prob_dir, order_metadata, 
    group_metadata, orders_by_model, free_margin, used_margin, curr_price_bid_ask,
     account_numbers, existing_order_amount, curr_spread, avg_prices, pair_bid_ask_map, 
     adjust_factor, strategy_weight, last_signal_update_time):

    base_model_key = pair

    buy_amount = 0
    sell_amount = 0
    float_profit = 0
    for order in orders:
        float_profit += order.PnL
        if order.dir:
            buy_amount += order.amount
        else:
            sell_amount += order.amount
    
    if base_model_key not in orders_by_model or len(orders) == 0:

        group_metadata[base_model_key + "_reduce_amount"] = 1

        if base_model_key + "_min_profit" in group_metadata:
            del group_metadata[base_model_key + "_min_profit"]

        if base_model_key + "_last_profit_udpate" in group_metadata:
            del group_metadata[base_model_key + "_last_profit_udpate"]

    if base_model_key + "_last_order_time" not in group_metadata:
        group_metadata[base_model_key + "_last_order_time"] = 0

    if base_model_key + "_last_dir" not in group_metadata:
        group_metadata[base_model_key + "_last_dir"] = None

    if base_model_key + "_last_order_dir" not in group_metadata:
        group_metadata[base_model_key + "_last_order_dir"] = None

    if base_model_key + "_last_signal_update_time" not in group_metadata:
        group_metadata[base_model_key + "_last_signal_update_time"] = None

    if len(orders) > 0:
        time_diff = calculate_time_diff(time.time(), group_metadata[base_model_key + "_last_order_time"])
    else:
        time_diff = 0

    if  group_metadata[base_model_key + "_last_signal_update_time"] != last_signal_update_time:
        # We close in opposite direction if positive, as next trade will not occur due to opposite trade direction to pos trade
        if len(orders) > 0 and float_profit > 0 and time_diff > 6:
            return True

    # We don't want to trade (add to) in same direction as losing trade
    if len(orders) > 0 and (prob_dir > 0) == group_metadata[base_model_key + "_last_order_dir"]:
        return False

    # We don't want to trade in the same direction as losing trade next time
    if len(orders) == 0 and ((prob_dir > 0) == group_metadata[base_model_key + "_last_dir"]):
        trade_logger.info('Last Signal Update') 
        return False

    if group_metadata[base_model_key + "_curr_spread"] >= 2.0:
        return False

    if group_metadata[base_model_key + "_last_signal_update_time"] == last_signal_update_time:
        trade_logger.info('Last Signal Update') 
        return False

    return True

def get_pair_prices(select_pair, group_metadata):

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
    
    ideal_spread = avg_spreads[select_pair]

    actual_spread = abs(pair_bid_ask_map[select_pair]['bid'] - pair_bid_ask_map[select_pair]['ask']) / pip_size
    print ("Actual Spread", actual_spread, pair_bid_ask_map[select_pair]['bid'], pair_bid_ask_map[select_pair]['ask'])
    actual_spread1 = actual_spread
    if actual_spread > 3.5:
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

    group_metadata[select_pair + "_curr_spread"] = curr_spread

    if curr_spread >= 2.0:
        trade_logger.info("Spread too big " + str(actual_spread1) + " " + str(ideal_spread))

    return avg_prices, pair_bid_ask_map

def enter_group_trades(pair, orders, growth_factor, prob_dir, order_metadata, group_metadata,
    orders_by_model, is_above_min_trade_volatility, free_margin, used_margin, account_numbers, 
    existing_order_amount, adjust_factor, 
    strategy_weight, last_signal_update_time, reward_risk_ratio, max_pip_barrier, 
    decay_frac, model_type, shap_values, is_filter_member_orders, is_filter_members_hedge):

    base_model_key = pair

    buy_amount = 0
    sell_amount = 0
    for order in orders:
        if order.dir:
            buy_amount += order.amount
        else:
            sell_amount += order.amount
    
    if base_model_key not in orders_by_model or len(orders) == 0:

        group_metadata[base_model_key + "_reduce_amount"] = 1

        if base_model_key + "_min_profit" in group_metadata:
            del group_metadata[base_model_key + "_min_profit"]

        if base_model_key + "_last_profit_udpate" in group_metadata:
            del group_metadata[base_model_key + "_last_profit_udpate"]

    if base_model_key + "_last_order_time" not in group_metadata:
        group_metadata[base_model_key + "_last_order_time"] = 0

    if base_model_key + "_last_dir" not in group_metadata:
        group_metadata[base_model_key + "_last_dir"] = None

    if base_model_key + "_last_order_dir" not in group_metadata:
        group_metadata[base_model_key + "_last_order_dir"] = None


    if is_valid_trading_period(time.time()) == False:
        return False

    #1. amount * (base / curr1) * (curr1 / curr2)
    #2. (curr2 / curr1) * (curr1 / base) 
    first_currency = pair[0:3]
    second_currency = pair[4:7]

    if "avg_prices_last_update" not in group_metadata:
        group_metadata["avg_prices_last_update"] = 0

    if time.time() - group_metadata["avg_prices_last_update"] > 60 * 60 * 24:
        avg_prices, pair_bid_ask_map = get_pair_prices(pair, group_metadata)
        group_metadata["avg_prices_last_update"] = time.time()
        group_metadata["avg_prices"] = avg_prices
        trade_logger.info("Update Pair Prices")

    if second_currency != "AUD":
        pair_mult = group_metadata["avg_prices"][second_currency + "_AUD"]
    else:
        pair_mult = 1.0

    if second_currency == "JPY":
        pair_mult *= 100

    new_order = Order()
    new_order.dir = (prob_dir > 0)

    if prob_dir > 0:
        order_amount = int(round((abs(prob_dir)) / (pair_mult)))
    else:
        order_amount = -int(round((abs(prob_dir)) / (pair_mult)))

    if order_amount == 0:
        trade_logger.info('Order Size Too Small ')
        return False

    if group_metadata[base_model_key + "_last_signal_update_time"] != last_signal_update_time:
        # We need to keep this here so we only update ideal position once for each signal update
        group_metadata[base_model_key + "_last_signal_update_time"] = last_signal_update_time 

        if model_type == ModelType.time_regression:

            mean_shap_vector = list(np.mean(np.array(shap_values), axis=0))
            norm = np.linalg.norm(mean_shap_vector, ord=2) 
            mean_shap_vector = [v / norm for v in mean_shap_vector]

            if (len(orders) == 0) or (base_model_key + "_mean_shap_vector" not in group_metadata):
                group_metadata[base_model_key + "_mean_shap_vector"] = mean_shap_vector
            else:
                dot_product = sum([a * b for a, b in zip(mean_shap_vector, group_metadata[base_model_key + "_mean_shap_vector"])])
                decay_frac =  (0.5 - (dot_product * 0.5)) 
                group_metadata[base_model_key + "_mean_shap_vector"] = [(a * decay_frac) + (b * (1 - decay_frac)) for a, b in zip(mean_shap_vector, group_metadata[base_model_key + "_mean_shap_vector"])]
                norm = np.linalg.norm(group_metadata[base_model_key + "_mean_shap_vector"], ord=2) 
                group_metadata[base_model_key + "_mean_shap_vector"] = [v / norm for v in group_metadata[base_model_key + "_mean_shap_vector"]]
                trade_logger.info('Dot Product: ' + str(dot_product) + ', Decay Frac: ' + str(decay_frac))

        if len(orders) > 0:
            group_metadata[base_model_key + "_ideal_order_size"] = (group_metadata[base_model_key + "_ideal_order_size"] * (1 - decay_frac)) + (decay_frac * order_amount)  
        else:
            group_metadata[base_model_key + "_ideal_order_size"] = order_amount

    if len(orders) > 0:
        ideal_position = group_metadata[base_model_key + "_ideal_order_size"]
        delta_pos = ideal_position - (buy_amount - sell_amount)
        delta_fraction = abs(delta_pos) / abs(buy_amount - sell_amount)
    else:
        delta_pos = order_amount
        ideal_position =  order_amount
        delta_fraction = 1.0

    trade_logger.info('Delta Position: ' + str(delta_pos))

    # We don't want to trade in the same direction as losing trade next time
    if len(orders) == 0 and ((ideal_position > 0) == group_metadata[base_model_key + "_last_dir"]):
        trade_logger.info('Prev Trade Signal: ' + str(group_metadata[base_model_key + "_last_dir"]))
        return False

    if is_above_min_trade_volatility == False:
        trade_logger.info('Min Trade Volatility ')
        return False

    if delta_fraction < 0.5:
        trade_logger.info('Delta Fraction Too Small ')
        return False

    order_amount = int(round(abs(delta_pos)))
    new_order.dir = delta_pos > 0

    if order_amount == 0:
        return False 

    if is_filter_member_orders and (is_filter_members_hedge == False):
        order_amount = adjust_orders(pair, orders, order_amount, new_order.dir, group_metadata)
        if order_amount == 0:
            trade_logger.info('Adjust Order No Delta ')
            return False

    trade_logger.info('Final Order Amount ' + str(order_amount))

    avg_prices, pair_bid_ask_map = get_pair_prices(pair, group_metadata)

    if group_metadata[base_model_key + "_curr_spread"] >= 2.0:
        trade_logger.info('Spread Too Big ')
        return False

    if delta_pos > 0:
        curr_price = pair_bid_ask_map[pair]['ask']
    else:
        curr_price = pair_bid_ask_map[pair]['bid']

    if second_currency == "JPY":
        barrier_size = (65 * growth_factor * adjust_factor) / (pair_mult * order_amount * 0.01)
        barrier_size = min(barrier_size, 0.01 * max_pip_barrier)
        tp_barrier = 0.01 * max_pip_barrier * reward_risk_ratio
    else:
        barrier_size = (65 * growth_factor * adjust_factor) / (pair_mult * order_amount)
        barrier_size = min(barrier_size , 0.0001 * max_pip_barrier)
        tp_barrier = 0.0001 * max_pip_barrier * reward_risk_ratio

    print growth_factor, pair_mult
    if new_order.dir:
        new_order.ideal_price = pair_bid_ask_map[pair]["ask"]
        stop_loss = new_order.ideal_price - barrier_size
        take_profit = new_order.ideal_price + (tp_barrier)
    else:
        new_order.ideal_price = pair_bid_ask_map[pair]["bid"]
        stop_loss = new_order.ideal_price + barrier_size
        take_profit = new_order.ideal_price - (tp_barrier)

    order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount, existing_order_amount, base_model_key, stop_loss, take_profit)

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
        group_metadata[base_model_key + "_growth_factor"] = growth_factor

        new_order.pair = pair
        new_order.order_id = order_id
        new_order.amount = order_amount
        new_order.account_number = account_number
        new_order.model_key = base_model_key
        orders.append(new_order)

        if is_filter_member_orders:
            group_metadata[base_model_key + "_orders"].append(order_id)

        trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 

        order_metadata = update_orders(orders, group_metadata, order_metadata)

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

def find_time_series_after_event(pair, news_time, time_diff_hours):

    prices, times = get_time_series(pair, (int(time_diff_hours) + 2) * 60, granularity = "M1")

    anchor_price = 0
    closest_time = abs(times[0] - news_time)
    for time, price in zip(times, prices):

        if abs(time - news_time) < closest_time:
            closest_time = abs(time - news_time)
            anchor_price = price


    prices, times = get_time_series(pair, (int(time_diff_hours) + 2))

    price_deltas = []
    for time, price in zip(times, prices):

        if len(price_deltas) >= time_diff_hours:
            break

        if time > news_time:
            price_deltas.append(price - anchor_price)


    return price_deltas[-1], anchor_price

def get_strategy_parameters(model_type):

    if model_type == ModelType.barrier:
        adjust_factors = {'EUR_USD_sample_num': 1132522, 'EUR_GBP_sample_num': 1381305, 'GBP_JPY_sample_num': 1300103, 'AUD_CAD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.4100263179578162, 'mean_sharpe': 0.11921492806180306, 'target': 0.6649162165279309, 'max_pip_barrier': 215.85330343019456, 'day_wait': 500000.0, 'max_order_size': 159.97556787508097, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_CAD', 'currency_weights': {}, 'min_trade_volatility': 57.92630410392942, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.1441170487333004, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 11.909556945478151, 'is_hedge': True, 'samples': 285.36474928727154}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.04242896132047754, 'target': 0.1830685368716719, 'min_trade_volatility': 81.79908978890846, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 100.0, 'reward_risk_ratio': 1.25, 'currency_pair': 'AUD_CAD', 'currency_weights': {'USD': 1.0, 'AUD': 0.27363341587382545, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'EUR': 1.0}, 'decay_frac': 0.5, 'is_low_barrier': False, 'auc_barrier': 1.831215657675268, 'sharpe': 0.1391568440493328, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 43.683994752061174, 'is_hedge': True, 'samples': 31.70719436525239}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.3466255191693878, 'mean_sharpe': 0.08871057394804872, 'target': 0.39295174392453475, 'max_pip_barrier': 185.08748544808526, 'day_wait': 500000.0, 'max_order_size': 159.9566698784891, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_CAD', 'currency_weights': {}, 'min_trade_volatility': 44.12533185932655, 'is_low_barrier': False, 'auc_barrier': 1.5225905853741968, 'sharpe': 0.10536131931608475, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 376.3239979875902}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.1, 'mean_sharpe': -0.0021242713627051316, 'target': 0.19447443980054488, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 167.6803598709909, 'reward_risk_ratio': 1.25, 'currency_pair': 'AUD_CAD', 'currency_weights': {'USD': 1.0, 'AUD': 0.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'EUR': 1.0}, 'min_trade_volatility': 77.98809114182863, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.1584610811468305, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.2033621230237994, 'max_barrier': 81.28663689501815, 'is_hedge': False, 'samples': 25.586114371960424}], 'USD_CAD_sample_num': 1197453, 'EUR_NZD': [{'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.1, 'mean_sharpe': 0.1689178780153617, 'target': 0.23016396582121262, 'max_pip_barrier': 219.88939751960152, 'day_wait': 500000.0, 'max_order_size': 178.13431608861407, 'reward_risk_ratio': 1.25, 'currency_pair': 'EUR_NZD', 'currency_weights': {}, 'min_trade_volatility': 122.5905568820461, 'is_low_barrier': False, 'auc_barrier': 0.1, 'sharpe': 0.18002898223943753, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 63.279126493485684, 'is_hedge': True, 'samples': 23.08407738095238}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 1.0, 'mean_sharpe': 0.025073710127160493, 'target': 0.14052771617959836, 'max_pip_barrier': 213.05846734891065, 'day_wait': 500000.0, 'max_order_size': 108.35612271793147, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_NZD', 'currency_weights': {}, 'min_trade_volatility': 50.6667287388681, 'is_low_barrier': False, 'auc_barrier': 0.6536072415377866, 'sharpe': 0.0847902914772267, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 59.718900470693654, 'is_hedge': True, 'samples': 158.3296130952381}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.37289396390208196, 'mean_sharpe': 0.06101979110775891, 'target': 0.135636967096906, 'max_pip_barrier': 151.18555884743753, 'day_wait': 500000.0, 'max_order_size': 170.20285955187288, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_NZD', 'currency_weights': {}, 'min_trade_volatility': 22.787967281482523, 'is_low_barrier': False, 'auc_barrier': 0.2801871084973984, 'sharpe': 0.08702517564469038, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 69.78708487716688, 'is_hedge': False, 'samples': 168.3779761904762}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.46567368286851274, 'mean_sharpe': 0.02012489170687468, 'target': 0.09191560964030371, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 40.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_NZD', 'currency_weights': {}, 'min_trade_volatility': 107.81909472748936, 'is_low_barrier': False, 'auc_barrier': 0.26628138957297637, 'sharpe': 0.08524974423906728, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 29.194568452380953}], 'EUR_GBP': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.6528108700952703, 'mean_sharpe': 0.18523078834628562, 'target': 0.40012278347723185, 'max_pip_barrier': 132.12748182203845, 'day_wait': 500000.0, 'max_order_size': 170.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_GBP', 'currency_weights': {}, 'min_trade_volatility': 76.39081896658223, 'is_low_barrier': False, 'auc_barrier': 0.800306722409814, 'sharpe': 0.23091532472197104, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 78.84880236778186, 'is_hedge': True, 'samples': 42.0253367543297}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.45024926821031036, 'mean_sharpe': 0.1587550757344405, 'target': 0.32430141266893536, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 71.57054118571138, 'reward_risk_ratio': 1.25, 'currency_pair': 'EUR_GBP', 'currency_weights': {}, 'min_trade_volatility': 62.337822679417066, 'is_low_barrier': False, 'auc_barrier': 1.1484666613827672, 'sharpe': 0.18971702130854462, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 91.63690194760166, 'is_hedge': True, 'samples': 51.27325208466966}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.3815724569144423, 'mean_sharpe': 0.1451203696625, 'target': 0.3050787109617764, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 146.7367563085018, 'reward_risk_ratio': 1.0281557572616389, 'currency_pair': 'EUR_GBP', 'currency_weights': {}, 'min_trade_volatility': 74.61013879205152, 'is_low_barrier': False, 'auc_barrier': 1.4662378237664258, 'sharpe': 0.20318984137410792, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 1.9836731162379473, 'max_barrier': 49.04822467926059, 'is_hedge': False, 'samples': 51.50737652341244}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.35716025959300424, 'mean_sharpe': 0.13081707493314707, 'target': 0.23634053215443196, 'max_pip_barrier': 93.60679053912362, 'day_wait': 500000.0, 'max_order_size': 157.25438816142577, 'reward_risk_ratio': 1.000000002288187, 'currency_pair': 'EUR_GBP', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.5112949642357484e-07, 'CAD': 1.0}, 'min_trade_volatility': 74.34652884769868, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.17291657105082028, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 30.670301475304683}], 'EUR_CHF': [{'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.5067748783228563, 'mean_sharpe': 0.10685923864201467, 'target': 0.6329555891560391, 'max_pip_barrier': 249.97959853835962, 'day_wait': 500000.0, 'max_order_size': 129.9004041271708, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_CHF', 'currency_weights': {}, 'min_trade_volatility': 23.005376924165002, 'is_low_barrier': False, 'auc_barrier': 1.5221182094063177, 'sharpe': 0.1357493910269179, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': True, 'samples': 427.5328172332548}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.2491603025254088, 'mean_sharpe': 0.15523958572848282, 'target': 0.39733751223308816, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 30.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_CHF', 'currency_weights': {}, 'min_trade_volatility': 76.03638787269767, 'is_low_barrier': False, 'auc_barrier': 1.8563604668181617, 'sharpe': 0.18760354654343503, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 92.8587712062383, 'is_hedge': True, 'samples': 135.95871199371703}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.5, 'mean_sharpe': 0.09233411411703256, 'target': 0.3699808619706602, 'max_pip_barrier': 197.6948986455448, 'day_wait': 500000.0, 'max_order_size': 71.18334890582538, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_CHF', 'currency_weights': {}, 'min_trade_volatility': 43.340261559303045, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.13187212286907216, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 65.08315491887048, 'is_hedge': False, 'samples': 392.9698193649725}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.5251347172178603, 'mean_sharpe': 0.06783715843866366, 'target': 0.18457350833170136, 'max_pip_barrier': 218.2340398610544, 'day_wait': 500000.0, 'max_order_size': 49.30364543783792, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_CHF', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 0.9998866851363514, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 52.45953972635215, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.11368583211347968, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.490643136358044, 'max_barrier': 46.690506997882906, 'is_hedge': False, 'samples': 164.2970941321665}], 'CAD_JPY': [{'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.0727263390664367, 'auc_barrier': 0.5010512065817109, 'min_trade_volatility': 60.52527491296848, 'max_pip_barrier': 100.12820491341402, 'day_wait': 500000.0, 'max_order_size': 91.16518237429098, 'reward_risk_ratio': 1.0, 'currency_pair': 'CAD_JPY', 'currency_weights': {}, 'decay_frac': 0.2334343844508786, 'target': 0.1358057309069577, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.09308549323617535, 'is_max_volatility': False, 'adjust_factor': 1.0887117132597672, 'max_barrier': 41.870129063413735, 'is_hedge': True, 'samples': 117.08126290038987}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.12069769235051642, 'auc_barrier': 1.4386141081864456, 'min_trade_volatility': 131.92264783094822, 'max_pip_barrier': 106.13140233340147, 'day_wait': 500000.0, 'max_order_size': 178.4911282854355, 'reward_risk_ratio': 1.0, 'currency_pair': 'CAD_JPY', 'currency_weights': {}, 'decay_frac': 0.5134055777841232, 'target': 0.2840701254050583, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.18589592396316632, 'is_max_volatility': False, 'adjust_factor': 1.0973258980633067, 'max_barrier': 40.01555942147463, 'is_hedge': True, 'samples': 44.310068037611806}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.04367129610394843, 'auc_barrier': 0.952644377437879, 'min_trade_volatility': 126.07271107194735, 'max_pip_barrier': 194.80962355249284, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'CAD_JPY', 'currency_weights': {}, 'decay_frac': 0.12848935325468994, 'target': 0.35623898722828234, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.14149402789131113, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 91.18721810259154}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.10498500410450666, 'target': 0.23074504205961616, 'min_trade_volatility': 126.03098213326443, 'max_pip_barrier': 80.14080725228646, 'day_wait': 500000.0, 'max_order_size': 139.98954136069318, 'reward_risk_ratio': 3.0, 'currency_pair': 'CAD_JPY', 'currency_weights': {}, 'decay_frac': 0.6368806195330966, 'is_low_barrier': False, 'auc_barrier': 1.1169476603069501, 'sharpe': 0.11806183853784206, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 57.368702698570445}], 'USD_CHF_sample_num': 1304934, 'NZD_JPY_sample_num': 1102996, 'AUD_JPY_sample_num': 1521737, 'default_day_wait': {False: 500000.0, True: 500000.0}, 'CAD_JPY_sample_num': 1704130, 'GBP_NZD_sample_num': 1337480, 'AUD_NZD': [{'max_order_size': 30.0, 'currency_weights': {}, 'is_relavent_currency': False, 'max_barrier': 65.95573807407082, 'is_any_barrier': False, 'mean_sharpe': 0.023623000929686402, 'decay_frac': 0.5, 'is_reverse_trade': False, 'samples': 143.88266315095584, 'sharpe': 0.07209994704519888, 'is_max_volatility': False, 'is_norm_signal': True, 'max_pip_barrier': 220.0, 'reward_risk_ratio': 1.25, 'currency_pair': 'AUD_NZD', 'is_close_pos_trade': False, 'auc_barrier': 0.7, 'is_hedge': True, 'target': 0.07840200155198106, 'day_wait': 5.369463411899454, 'min_trade_volatility': 50, 'is_low_barrier': False, 'adjust_factor': 0.8}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 1.0, 'mean_sharpe': 0.1369406203169827, 'target': 0.2308889480954914, 'max_pip_barrier': 158.1119975947118, 'day_wait': 500000.0, 'max_order_size': 153.14500017502507, 'reward_risk_ratio': 1.0236786258423485, 'currency_pair': 'AUD_NZD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 0.9995969971046181, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 93.35221475380601, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.18509805582397318, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.4830398490474552, 'max_barrier': 100.0, 'is_hedge': True, 'samples': 25.365493022825987}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.6899159342816044, 'mean_sharpe': 0.17696629138366005, 'target': 0.26052188275436816, 'max_pip_barrier': 198.84748308804336, 'day_wait': 500000.0, 'max_order_size': 169.0160648328747, 'reward_risk_ratio': 1.25, 'currency_pair': 'AUD_NZD', 'currency_weights': {}, 'min_trade_volatility': 128.86235336424159, 'is_low_barrier': False, 'auc_barrier': 1.9868262254604203, 'sharpe': 0.24675037387177398, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 29.10356184595412, 'is_hedge': False, 'samples': 11.06367248867942}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.17780735891752214, 'mean_sharpe': -0.012945583941290447, 'target': 0.06234239136865522, 'max_pip_barrier': 118.4832395847071, 'day_wait': 500000.0, 'max_order_size': 40.000000000000014, 'reward_risk_ratio': 1.0397478905730557, 'currency_pair': 'AUD_NZD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 0.7059645005460237, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 50.0, 'is_low_barrier': False, 'auc_barrier': 0.7, 'sharpe': 0.060764621630869306, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.485415456309928, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 23.88134183532021}], 'GBP_AUD': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': -0.008227750157032407, 'auc_barrier': 1.6685241446878005, 'min_trade_volatility': 91.42212650816607, 'max_pip_barrier': 234.98264751332616, 'day_wait': 500000.0, 'max_order_size': 40.68621149986953, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_AUD', 'currency_weights': {}, 'decay_frac': 0.9021171312685543, 'target': 0.07347042429080455, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.038504982100778025, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.026314731527464, 'is_hedge': True, 'samples': 505.6185473646079}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.033728220570105216, 'auc_barrier': 0.24701041463155898, 'min_trade_volatility': 145.0835860481958, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 2.9980555126322286, 'currency_pair': 'GBP_AUD', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 0.09632856117765387, 'CHF': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'GBP': 1.0, 'EUR': 1.0}, 'decay_frac': 1.0, 'target': 0.1728912699046971, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.14441480875390098, 'is_max_volatility': False, 'adjust_factor': 2.4969992829611165, 'max_barrier': 40.0, 'is_hedge': True, 'samples': 14.676689005614925}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.020667887804320485, 'auc_barrier': 2.0, 'min_trade_volatility': 150.0, 'max_pip_barrier': 139.2946448620428, 'day_wait': 500000.0, 'max_order_size': 91.32968932954641, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_AUD', 'currency_weights': {}, 'decay_frac': 0.9999950382850862, 'target': 0.05819746411373166, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.033764100885330245, 'is_max_volatility': True, 'adjust_factor': 2.49999999999973, 'max_barrier': 96.90221806886112, 'is_hedge': False, 'samples': 258.3626154682123}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.024965588283298413, 'target': 0.054795278977149156, 'min_trade_volatility': 92.3547770274584, 'max_pip_barrier': 105.35242642247216, 'day_wait': 500000.0, 'max_order_size': 179.9244593499744, 'reward_risk_ratio': 3.0, 'currency_pair': 'GBP_AUD', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'GBP': 0.5122379771277821, 'EUR': 1.0}, 'decay_frac': 0.5103956975048489, 'is_low_barrier': False, 'auc_barrier': 0.46079206279329177, 'sharpe': 0.045339324623353, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 76.99686572525667, 'is_hedge': False, 'samples': 79.59789893135301}], 'GBP_CHF': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.20993662993481466, 'target': 2.8865228578727242, 'min_trade_volatility': 53.25975440450817, 'max_pip_barrier': 216.14702306505046, 'day_wait': 500000.0, 'max_order_size': 50.692811975367626, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_CHF', 'currency_weights': {}, 'decay_frac': 0.9714186335276075, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.2244696145144719, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 54.22963662800757, 'is_hedge': True, 'samples': 616.8361681501718}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.07866673185851515, 'auc_barrier': 1.5185992436259794, 'min_trade_volatility': 11.428262029681163, 'max_pip_barrier': 219.93822332256536, 'day_wait': 500000.0, 'max_order_size': 89.66083146100921, 'reward_risk_ratio': 1.5, 'currency_pair': 'GBP_CHF', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 1.0, 'CHF': 0.7978975040037272, 'NZD': 1.0, 'EUR': 1.0, 'GBP': 1.0, 'CAD': 1.0}, 'decay_frac': 0.6460190219254527, 'target': 0.3245129304126295, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.09841777520982578, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': True, 'samples': 263.6414911430334}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.15112872256770046, 'auc_barrier': 2.0, 'min_trade_volatility': 10.000000000000007, 'max_pip_barrier': 233.97678045337923, 'day_wait': 500000.0, 'max_order_size': 44.10101100266774, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_CHF', 'currency_weights': {}, 'decay_frac': 0.9573548318187648, 'target': 1.5249310036331831, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.16978781476428992, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 61.87227369141274, 'is_hedge': False, 'samples': 671.9062307217766}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.15565606116118472, 'target': 0.5650920181075656, 'min_trade_volatility': 100.19716189240197, 'max_pip_barrier': 219.85025727023856, 'day_wait': 500000.0, 'max_order_size': 30.883371611083227, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_CHF', 'currency_weights': {}, 'decay_frac': 1.0, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.18558523356903756, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 63.9198897926559, 'is_hedge': False, 'samples': 189.91451484974002}], 'NZD_CHF': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.03916671551064667, 'target': 0.14105977391317354, 'min_trade_volatility': 150.0, 'max_pip_barrier': 210.40912814329724, 'day_wait': 500000.0, 'max_order_size': 117.2343920285971, 'reward_risk_ratio': 1.25, 'currency_pair': 'NZD_CHF', 'currency_weights': {}, 'decay_frac': 0.6209879044163914, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.10818150135348094, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 57.5334767787904, 'is_hedge': True, 'samples': 45.788307482529405}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.003260482330040089, 'auc_barrier': 1.0467968270303545, 'min_trade_volatility': 55.6679503622357, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 136.18328852459854, 'reward_risk_ratio': 1.5, 'currency_pair': 'NZD_CHF', 'currency_weights': {}, 'decay_frac': 0.9999750951595016, 'target': 0.02724340428301051, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.026318893378569075, 'is_max_volatility': False, 'adjust_factor': 0.5, 'max_barrier': 68.86023429651603, 'is_hedge': True, 'samples': 42.55326401908982}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.0637435787484438, 'target': 0.10099089662762493, 'min_trade_volatility': 50.0, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 30.000000000000014, 'reward_risk_ratio': 1.0370679157046714, 'currency_pair': 'NZD_CHF', 'currency_weights': {}, 'decay_frac': 0.7716822778259576, 'is_low_barrier': False, 'auc_barrier': 0.7, 'sharpe': 0.09600019635517429, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 11.695803563627774, 'is_hedge': False, 'samples': 38.82052156127493}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.15665913440427576, 'target': 0.22997679689149383, 'min_trade_volatility': 72.84810232018636, 'max_pip_barrier': 200.1273022234687, 'day_wait': 500000.0, 'max_order_size': 178.0038856005128, 'reward_risk_ratio': 1.25, 'currency_pair': 'NZD_CHF', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 0.652286817051423, 'CAD': 1.0, 'GBP': 1.0, 'EUR': 1.0}, 'decay_frac': 0.41984703593076517, 'is_low_barrier': False, 'auc_barrier': 1.0835095570711655, 'sharpe': 0.1936424849379703, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 50.404251542724396, 'is_hedge': False, 'samples': 26.004772456110448}], 'CHF_JPY': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.3504635467640387, 'mean_sharpe': 0.16266039028573456, 'target': 1.3779421058127028, 'max_pip_barrier': 249.94307767316656, 'day_wait': 500000.0, 'max_order_size': 73.86740824974667, 'reward_risk_ratio': 1.0, 'currency_pair': 'CHF_JPY', 'currency_weights': {}, 'min_trade_volatility': 43.608403511531755, 'is_low_barrier': False, 'auc_barrier': 1.612227557798167, 'sharpe': 0.1999859201639729, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 86.94113036435769, 'is_hedge': True, 'samples': 511.800203412815}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.7325313727283767, 'mean_sharpe': 0.07021690762571661, 'target': 0.19703578092107996, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 107.3937362389129, 'reward_risk_ratio': 1.0, 'currency_pair': 'CHF_JPY', 'currency_weights': {}, 'min_trade_volatility': 95.09618384266909, 'is_low_barrier': False, 'auc_barrier': 0.7148035654078118, 'sharpe': 0.15620811079765162, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 91.03931279290282, 'is_hedge': True, 'samples': 30.52322296304667}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.886551325997319, 'mean_sharpe': 0.13269772627665682, 'target': 0.8490809822869511, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'CHF_JPY', 'currency_weights': {}, 'min_trade_volatility': 65.91238709170668, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.17821985859831743, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 382.7777149960447}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.7357178312785773, 'mean_sharpe': 0.23731190814920952, 'target': 0.39468583796422074, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 131.06834450866248, 'reward_risk_ratio': 1.0, 'currency_pair': 'CHF_JPY', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 0.5150239395148799, 'GBP': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'EUR': 1.0}, 'min_trade_volatility': 115.06790689120457, 'is_low_barrier': False, 'auc_barrier': 1.1623453360865146, 'sharpe': 0.2837856927777655, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 79.66989024521838, 'is_hedge': False, 'samples': 32.338117301389985}], 'CAD_CHF_sample_num': 1604939, 'AUD_CHF_sample_num': 1323562, 'EUR_JPY_sample_num': 1919516, 'AUD_CHF': [{'max_order_size': 134.21590970696417, 'currency_weights': {}, 'is_relavent_currency': False, 'max_barrier': 59.68640135348222, 'is_any_barrier': False, 'mean_sharpe': 0.16722786342827684, 'decay_frac': 0.5, 'samples': 146.27174880634456, 'sharpe': 0.23509078617604484, 'is_max_volatility': False, 'is_norm_signal': False, 'max_pip_barrier': 219.18790035771025, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_CHF', 'auc_barrier': 2.0, 'is_hedge': True, 'trade_volatility_factor': 0.1, 'target': 1.0213168467431328, 'day_wait': 500000.0, 'min_trade_volatility': 106.90795206863737, 'is_low_barrier': False, 'adjust_factor': 2.5}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.1, 'mean_sharpe': 0.18937373546491268, 'target': 0.29605361504465677, 'max_pip_barrier': 209.488465937032, 'day_wait': 500000.0, 'max_order_size': 159.90787818618, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_CHF', 'currency_weights': {'USD': 1.0, 'AUD': 0.6179235988164428, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 82.54510966112275, 'is_low_barrier': False, 'auc_barrier': 0.9867045300070113, 'sharpe': 0.25345941909491154, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 38.88192541894843, 'is_hedge': True, 'samples': 18.904264789188314}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.5, 'mean_sharpe': 0.14789179248993145, 'target': 0.7575256429935548, 'max_pip_barrier': 225.34495369580046, 'day_wait': 500000.0, 'max_order_size': 87.89166941777708, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_CHF', 'currency_weights': {}, 'min_trade_volatility': 109.57538201494172, 'is_low_barrier': False, 'auc_barrier': 1.8618681600451477, 'sharpe': 0.2132550296492488, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 70.57435672072921, 'is_hedge': False, 'samples': 137.88298130614226}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.19617169362238734, 'mean_sharpe': 0.259034585296173, 'target': 0.4299445388841061, 'max_pip_barrier': 219.70470905905415, 'day_wait': 500000.0, 'max_order_size': 179.20954523412192, 'reward_risk_ratio': 1.4627752578617454, 'currency_pair': 'AUD_CHF', 'currency_weights': {}, 'min_trade_volatility': 77.409184258739, 'is_low_barrier': False, 'auc_barrier': 0.49919781106982364, 'sharpe': 0.3262462208036093, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 59.94763299749947, 'is_hedge': False, 'samples': 12.40592376790483}], 'EUR_USD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.7755271706547926, 'mean_sharpe': 0.09585896366589845, 'target': 0.1916324624865937, 'max_pip_barrier': 129.05503513671374, 'day_wait': 500000.0, 'max_order_size': 40.0, 'reward_risk_ratio': 3.0, 'currency_pair': 'EUR_USD', 'currency_weights': {}, 'min_trade_volatility': 102.33762651326569, 'is_low_barrier': False, 'auc_barrier': 0.1, 'sharpe': 0.1746717724087873, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': True, 'samples': 12.292384232682739}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.015203254005555078, 'target': 0.05316242456235007, 'min_trade_volatility': 144.72091096897756, 'max_pip_barrier': 216.51825696453176, 'day_wait': 500000.0, 'max_order_size': 40.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_USD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 0.0, 'CAD': 1.0}, 'decay_frac': 0.679630651623584, 'is_low_barrier': False, 'auc_barrier': 1.0925195839978887, 'sharpe': 0.04977162132879048, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 72.28276599763127, 'is_hedge': True, 'samples': 44.367909238249595}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.1601788902773589, 'target': 0.21033803735512718, 'min_trade_volatility': 137.2692326824573, 'max_pip_barrier': 166.3319720816614, 'day_wait': 500000.0, 'max_order_size': 65.61453062815674, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_USD', 'currency_weights': {}, 'decay_frac': 0.7040415190147888, 'is_low_barrier': False, 'auc_barrier': 0.38918377391222314, 'sharpe': 0.20852436111101147, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 10.260698923028711, 'is_hedge': False, 'samples': 11.831442463533225}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': -0.03752295208281612, 'target': -0.0331116026566986, 'min_trade_volatility': 40.30122030077458, 'max_pip_barrier': 249.99735633156735, 'day_wait': 500000.0, 'max_order_size': 108.52080480551278, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_USD', 'currency_weights': {'USD': 0.2688460557762535, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'decay_frac': 0.7699515221740693, 'is_low_barrier': False, 'auc_barrier': 0.1479966610664607, 'sharpe': -0.0331116026566986, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 1.334609604734254, 'max_barrier': 80.54105819329035, 'is_hedge': False, 'samples': 11.831442463533225}], 'GBP_AUD_sample_num': 1440023, 'GBP_NZD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.7591754393131015, 'mean_sharpe': 0.05041936493176937, 'target': 0.0949858908278892, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 115.22207781380055, 'reward_risk_ratio': 1.1178934630846526, 'currency_pair': 'GBP_NZD', 'currency_weights': {}, 'min_trade_volatility': 104.47172928931079, 'is_low_barrier': False, 'auc_barrier': 0.10258486930732301, 'sharpe': 0.06824815186923107, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.3013947569477713, 'max_barrier': 64.05681368090845, 'is_hedge': True, 'samples': 64.6231841669104}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.039297929162674126, 'target': 0.07436880778515052, 'max_pip_barrier': 100.04350879614746, 'day_wait': 500000.0, 'max_order_size': 77.5010132801099, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_NZD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 0.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 131.0012979103459, 'is_low_barrier': False, 'auc_barrier': 0.41756109876333525, 'sharpe': 0.06806931318046017, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.01754430457459, 'is_hedge': True, 'samples': 56.5097006922102}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.17753917903269356, 'mean_sharpe': 0.03785065185045281, 'target': 0.10716793900897333, 'max_pip_barrier': 219.56149694954902, 'day_wait': 500000.0, 'max_order_size': 30.287057552060897, 'reward_risk_ratio': 1.3007932996307732, 'currency_pair': 'GBP_NZD', 'currency_weights': {}, 'min_trade_volatility': 135.51277953444077, 'is_low_barrier': False, 'auc_barrier': 0.8819230772222697, 'sharpe': 0.06874914327430284, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 23.50280384516458, 'is_hedge': False, 'samples': 322.8312372038608}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.06068693571999651, 'target': 0.06480135364487362, 'max_pip_barrier': 99.66763881032185, 'day_wait': 500000.0, 'max_order_size': 54.398098470616205, 'reward_risk_ratio': 1.2873711526333815, 'currency_pair': 'GBP_NZD', 'currency_weights': {}, 'min_trade_volatility': 138.82233224179976, 'is_low_barrier': False, 'auc_barrier': 0.24836813244693087, 'sharpe': 0.061803670461057045, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 0.5, 'max_barrier': 40.00000001463903, 'is_hedge': False, 'samples': 33.877352052256995}], 'NZD_JPY': [{'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.14480403881147302, 'auc_barrier': 2.0, 'min_trade_volatility': 111.12521275743399, 'max_pip_barrier': 219.67896769292832, 'day_wait': 500000.0, 'max_order_size': 60.303794043133045, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_JPY', 'currency_weights': {}, 'decay_frac': 0.5, 'target': 0.46603249745812214, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.17262198175256338, 'is_max_volatility': True, 'adjust_factor': 2.1738833119487486, 'max_barrier': 95.57517917618205, 'is_hedge': True, 'samples': 201.08527131782944}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.11412327425460524, 'target': 0.23576568531370345, 'min_trade_volatility': 114.87534656299219, 'max_pip_barrier': 244.83616572319744, 'day_wait': 500000.0, 'max_order_size': 112.98620489739382, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_JPY', 'currency_weights': {'JPY': 0.4753462848899065, 'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'GBP': 1.0, 'CAD': 1.0}, 'decay_frac': 0.362490532981938, 'is_low_barrier': False, 'auc_barrier': 1.4812163221084926, 'sharpe': 0.1520241773556402, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 7.63236985465211, 'is_hedge': True, 'samples': 67.90697674418604}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.09257859823590818, 'auc_barrier': 0.6979223090738336, 'min_trade_volatility': 102.18629433312844, 'max_pip_barrier': 190.56567826294057, 'day_wait': 500000.0, 'max_order_size': 168.44887425702166, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_JPY', 'currency_weights': {}, 'decay_frac': 0.5, 'target': 0.2038784342915877, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.11641989207434716, 'is_max_volatility': False, 'adjust_factor': 2.4096998222892116, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 148.9922480620155}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.16413622565914465, 'auc_barrier': 0.8557151177839551, 'min_trade_volatility': 141.41543406142435, 'max_pip_barrier': 249.9243814796305, 'day_wait': 500000.0, 'max_order_size': 65.99606117594126, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_JPY', 'currency_weights': {'JPY': 0.4946809126649242, 'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'GBP': 1.0, 'CAD': 1.0}, 'decay_frac': 0.19332170917049263, 'target': 0.22020944070536072, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.17747296708725419, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 55.16921443537733, 'is_hedge': False, 'samples': 25.58139534883721}], 'EUR_CAD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.4645512018782161, 'mean_sharpe': -0.0014219026263301811, 'target': 0.022690216030674556, 'max_pip_barrier': 198.14657092580347, 'day_wait': 500000.0, 'max_order_size': 118.14681411778993, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_CAD', 'currency_weights': {}, 'min_trade_volatility': 50.0, 'is_low_barrier': False, 'auc_barrier': 0.6, 'sharpe': 0.0179533517348447, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.1612657667484125, 'max_barrier': 30.65634979229501, 'is_hedge': True, 'samples': 320.08123833571193}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.4809254937338294, 'mean_sharpe': 0.006870489793403819, 'target': 0.06296578276986543, 'max_pip_barrier': 219.82674388055068, 'day_wait': 500000.0, 'max_order_size': 102.3944512800189, 'reward_risk_ratio': 1.5, 'currency_pair': 'EUR_CAD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 0.9322671779265775}, 'min_trade_volatility': 33.627114190374186, 'is_low_barrier': False, 'auc_barrier': 1.5309535304363573, 'sharpe': 0.05299622852052678, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.4707511907249295, 'max_barrier': 20.962176866268777, 'is_hedge': True, 'samples': 161.72357009550996}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.3643003402176776, 'mean_sharpe': 0.0006682799045337895, 'target': 0.05731573006715634, 'max_pip_barrier': 219.5452939356664, 'day_wait': 500000.0, 'max_order_size': 76.74228153141365, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_CAD', 'currency_weights': {}, 'min_trade_volatility': 14.379772666992006, 'is_low_barrier': False, 'auc_barrier': 0.4629747049400825, 'sharpe': 0.052783248377889704, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 23.28249636080779, 'is_hedge': False, 'samples': 72.28674936875618}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.27291010079685063, 'mean_sharpe': -0.008379760282646248, 'target': 0.08835908085322278, 'max_pip_barrier': 189.34257258058423, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 1.25, 'currency_pair': 'EUR_CAD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 0.0}, 'min_trade_volatility': 87.15801705931564, 'is_low_barrier': False, 'auc_barrier': 0.861321078003779, 'sharpe': 0.07718866373819061, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 1.5343977025401307, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 67.1577560654298}], 'EUR_CAD_sample_num': 1154969, 'CAD_CHF': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': -0.01036370050631264, 'auc_barrier': 1.5744784370642337, 'min_trade_volatility': 133.74306896916997, 'max_pip_barrier': 218.45773757994263, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'CAD_CHF', 'currency_weights': {}, 'decay_frac': 0.23942726693250685, 'target': 0.0639026508297021, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.05357031179533755, 'is_max_volatility': False, 'adjust_factor': 2.499999896965784, 'max_barrier': 40.0, 'is_hedge': True, 'samples': 69.03408630880091}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.1445150593730104, 'target': 0.1574934492389066, 'min_trade_volatility': 140.15197098395083, 'max_pip_barrier': 80.0, 'day_wait': 500000.0, 'max_order_size': 30.0, 'reward_risk_ratio': 3.0, 'currency_pair': 'CAD_CHF', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 1.0, 'CHF': 0.38604145756578806, 'NZD': 1.0, 'CAD': 1.0, 'GBP': 1.0, 'EUR': 1.0}, 'decay_frac': 0.5989128033005637, 'is_low_barrier': False, 'auc_barrier': 0.5940176327814297, 'sharpe': 0.15004900155267062, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 0.5, 'max_barrier': 47.790310628276394, 'is_hedge': True, 'samples': 10.638814670876853}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': -0.04424817639416124, 'auc_barrier': 1.9433905260129372, 'min_trade_volatility': 128.12462625479813, 'max_pip_barrier': 249.87115112083043, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'CAD_CHF', 'currency_weights': {}, 'decay_frac': 0.5237201293930205, 'target': 0.032126045927847985, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.029197469643956427, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 79.7911100315764}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.10688662076283913, 'auc_barrier': 0.48079348576480946, 'min_trade_volatility': 78.59577033181858, 'max_pip_barrier': 80.0, 'day_wait': 500000.0, 'max_order_size': 40.0, 'reward_risk_ratio': 2.9879952696187932, 'currency_pair': 'CAD_CHF', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 1.0, 'CHF': 0.20018209511070081, 'NZD': 1.0, 'CAD': 1.0, 'GBP': 1.0, 'EUR': 1.0}, 'decay_frac': 0.24423713788318835, 'target': 0.14458864620808926, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.1381477302777686, 'is_max_volatility': False, 'adjust_factor': 0.5108091484169635, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 17.849566836693384}], 'GBP_CAD_sample_num': 1008244, 'AUD_USD_sample_num': 998414, 'AUD_USD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.5246291952922036, 'mean_sharpe': 0.10482995357548967, 'target': 0.9063116872369843, 'max_pip_barrier': 232.41774840849152, 'day_wait': 500000.0, 'max_order_size': 126.66719690528065, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_USD', 'currency_weights': {}, 'min_trade_volatility': 10.0, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.12819640616536188, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 87.73967824626435, 'is_hedge': True, 'samples': 672.4487998238274}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.7873573944814279, 'mean_sharpe': 0.008337996852898565, 'target': 0.06168403238049994, 'max_pip_barrier': 249.91097953083386, 'day_wait': 500000.0, 'max_order_size': 152.28849945346877, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_USD', 'currency_weights': {}, 'min_trade_volatility': 81.23580811455338, 'is_low_barrier': False, 'auc_barrier': 0.737213012801998, 'sharpe': 0.055695205550106944, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.0, 'is_hedge': True, 'samples': 47.423475005505395}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.9644961294862564, 'mean_sharpe': 0.07252974879795761, 'target': 0.40546894132942307, 'max_pip_barrier': 245.42509664286555, 'day_wait': 500000.0, 'max_order_size': 134.25666736682297, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_USD', 'currency_weights': {}, 'min_trade_volatility': 24.923479100028697, 'is_low_barrier': False, 'auc_barrier': 1.2638231144377126, 'sharpe': 0.09978629396855043, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 91.72622208445443, 'is_hedge': False, 'samples': 505.4217132790135}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.10000000217614091, 'mean_sharpe': 0.024920012613080054, 'target': 0.05284504216318977, 'max_pip_barrier': 218.47024714768517, 'day_wait': 500000.0, 'max_order_size': 39.06799442290322, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_USD', 'currency_weights': {}, 'min_trade_volatility': 56.50537336637351, 'is_low_barrier': False, 'auc_barrier': 1.314584045491632, 'sharpe': 0.04327705509679543, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.424542085139791, 'max_barrier': 71.71081494624922, 'is_hedge': False, 'samples': 161.4005725611099}], 'default_adjust': {False: 2.394850774809067, True: 2.0439466648092752}, 'USD_CAD': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': -0.006357048863907665, 'target': 0.04334312921134925, 'min_trade_volatility': 105.6959069795891, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 180.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CAD', 'currency_weights': {}, 'decay_frac': 0.446377970075892, 'is_low_barrier': False, 'auc_barrier': 0.9968172454759212, 'sharpe': 0.03469704935688715, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 71.5575764762132, 'is_hedge': True, 'samples': 147.68937134677324}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.0481885160767759, 'target': 0.07455135628673794, 'min_trade_volatility': 28.74310480008336, 'max_pip_barrier': 226.62913285769162, 'day_wait': 500000.0, 'max_order_size': 170.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CAD', 'currency_weights': {'USD': 0.02548199492025589, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'EUR': 1.0}, 'decay_frac': 0.5016877472386386, 'is_low_barrier': False, 'auc_barrier': 0.8826526094887721, 'sharpe': 0.07122826081547076, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 0.5, 'max_barrier': 41.57807612330028, 'is_hedge': True, 'samples': 102.58141476798282}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.006750331014538106, 'target': 0.04565016146919503, 'min_trade_volatility': 134.4405170495061, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 100.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CAD', 'currency_weights': {}, 'decay_frac': 0.47204817934000315, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.043030379582257125, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 0.5000000017049093, 'max_barrier': 66.69652727223804, 'is_hedge': False, 'samples': 88.99677919599189}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.06101416822343487, 'target': 0.10325626003759668, 'min_trade_volatility': 20.0, 'max_pip_barrier': 185.64364862422212, 'day_wait': 500000.0, 'max_order_size': 99.22933639245979, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CAD', 'currency_weights': {}, 'decay_frac': 0.35266735921196335, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.09487977451946392, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 0.5, 'max_barrier': 16.16914680820235, 'is_hedge': False, 'samples': 139.50375760467614}], 'AUD_CAD_sample_num': 1270945, 'GBP_JPY': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.9999999999999999, 'mean_sharpe': 0.18467470351600335, 'target': 0.2630311709952878, 'max_pip_barrier': 249.92161013801876, 'day_wait': 500000.0, 'max_order_size': 84.13839980148882, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_JPY', 'currency_weights': {}, 'min_trade_volatility': 150.0, 'is_low_barrier': False, 'auc_barrier': 0.1, 'sharpe': 0.21330825671801507, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 50.56166247631626, 'is_hedge': True, 'samples': 20.85126374142629}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.316375917257556, 'mean_sharpe': -0.011152176362128096, 'target': 0.030491821231210137, 'max_pip_barrier': 194.61620317057935, 'day_wait': 500000.0, 'max_order_size': 65.90710394667173, 'reward_risk_ratio': 1.25, 'currency_pair': 'GBP_JPY', 'currency_weights': {}, 'min_trade_volatility': 150.0, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.024978170411663458, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 1.4455701542080726, 'max_barrier': 45.99448723263962, 'is_hedge': True, 'samples': 219.48698675185568}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.1, 'mean_sharpe': 0.08924322417452517, 'target': 0.18854713552851876, 'max_pip_barrier': 249.94060517793454, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_JPY', 'currency_weights': {}, 'min_trade_volatility': 141.47019839163184, 'is_low_barrier': False, 'auc_barrier': 0.1, 'sharpe': 0.1670301173521184, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 17.421779573428545}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.1000001623175326, 'mean_sharpe': 0.025859258595941268, 'target': 0.08432719583551483, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 82.19199360037392, 'reward_risk_ratio': 1.25, 'currency_pair': 'GBP_JPY', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 0.7011926208255327, 'NZD': 1.0, 'CAD': 1.0, 'EUR': 1.0}, 'min_trade_volatility': 50.0, 'is_low_barrier': False, 'auc_barrier': 0.7, 'sharpe': 0.0526682345037437, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 1.593501969171653, 'max_barrier': 95.44452674893724, 'is_hedge': False, 'samples': 213.72545334961947}], 'AUD_JPY': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.07253267905006258, 'target': 0.6794943637090589, 'max_pip_barrier': 250.0, 'day_wait': 500000.0, 'max_order_size': 152.4000640720446, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_JPY', 'currency_weights': {}, 'min_trade_volatility': 10.0, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.10394604347900557, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.0, 'is_hedge': True, 'samples': 595.194553923325}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.1, 'mean_sharpe': 0.016543141977142472, 'target': 0.09517923129521243, 'max_pip_barrier': 181.23795449755636, 'day_wait': 500000.0, 'max_order_size': 30.040017354290985, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_JPY', 'currency_weights': {'USD': 1.0, 'AUD': 0.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 56.149365661323415, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.0565447780713349, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 84.90184632916001, 'is_hedge': True, 'samples': 183.297742744536}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.9064592450461462, 'mean_sharpe': 0.04809304516086607, 'target': 0.379877869118086, 'max_pip_barrier': 244.0926100605696, 'day_wait': 500000.0, 'max_order_size': 130.21407449584484, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_JPY', 'currency_weights': {}, 'min_trade_volatility': 10.061211871843398, 'is_low_barrier': False, 'auc_barrier': 1.9955932349573309, 'sharpe': 0.06729574367813373, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 74.15347566719223, 'is_hedge': False, 'samples': 676.6950913651021}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.19745206464489198, 'mean_sharpe': 0.012396798037984727, 'target': 0.1364648570861432, 'max_pip_barrier': 219.29554631665087, 'day_wait': 500000.0, 'max_order_size': 144.02986532677653, 'reward_risk_ratio': 1.0, 'currency_pair': 'AUD_JPY', 'currency_weights': {'USD': 1.0, 'AUD': 9.401460716510077e-08, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 69.58922393068747, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.08226273084536202, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 50.83281634595012, 'is_hedge': False, 'samples': 135.90397706915084}], 'EUR_AUD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.18321836869925562, 'target': 1.0368676434540105, 'max_pip_barrier': 222.5656163393706, 'day_wait': 500000.0, 'max_order_size': 98.26966692330842, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_AUD', 'currency_weights': {}, 'min_trade_volatility': 150.0, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.22508128167419972, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 40.68363909309953, 'is_hedge': True, 'samples': 154.37704918032787}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.1638544384683618, 'target': 0.740647729615788, 'max_pip_barrier': 249.96832299405762, 'day_wait': 500000.0, 'max_order_size': 82.30359792894497, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_AUD', 'currency_weights': {}, 'min_trade_volatility': 150.0, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.21508135312415036, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 52.84829691695434, 'is_hedge': True, 'samples': 103.24441132637854}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.05811087817272093, 'target': 0.4054944004427504, 'max_pip_barrier': 151.22299589724634, 'day_wait': 500000.0, 'max_order_size': 179.892347492218, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_AUD', 'currency_weights': {}, 'min_trade_volatility': 49.576657036766285, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.07073274404519325, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 56.5316934151753, 'is_hedge': False, 'samples': 671.1430700447094}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.0963130693118896, 'target': 0.21207184192875747, 'max_pip_barrier': 224.37835163465726, 'day_wait': 500000.0, 'max_order_size': 60.00816873895656, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_AUD', 'currency_weights': {'USD': 1.0, 'AUD': 0.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 142.46497299730848, 'is_low_barrier': False, 'auc_barrier': 1.2350242676715815, 'sharpe': 0.11629043953920129, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 62.89984084330464, 'is_hedge': False, 'samples': 82.68256333830105}], 'USD_JPY': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.07812717640985108, 'target': 0.4777792407673036, 'min_trade_volatility': 58.78353745736093, 'max_pip_barrier': 214.2389780002992, 'day_wait': 500000.0, 'max_order_size': 166.2684409411509, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_JPY', 'currency_weights': {}, 'decay_frac': 0.520761717963373, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.10694995777685602, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 67.20583899156676, 'is_hedge': True, 'samples': 374.7350190509179}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.5, 'mean_sharpe': 0.03911405115970961, 'target': 0.14509251935110698, 'max_pip_barrier': 176.49831563088404, 'day_wait': 500000.0, 'max_order_size': 160.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_JPY', 'currency_weights': {}, 'min_trade_volatility': 54.958404023008924, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.07803930034342478, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 53.704359791596396, 'is_hedge': True, 'samples': 203.42396951853135}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.08470796060384038, 'target': 0.33970174530402425, 'min_trade_volatility': 86.46982561840188, 'max_pip_barrier': 209.03847574706506, 'day_wait': 500000.0, 'max_order_size': 169.1080359438542, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_JPY', 'currency_weights': {}, 'decay_frac': 0.6595105219275409, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.1309064248089282, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 77.40925735219088, 'is_hedge': False, 'samples': 163.472462764115}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.03855454062723959, 'auc_barrier': 1.5488199066949997, 'min_trade_volatility': 65.36110616174619, 'max_pip_barrier': 194.76535943680705, 'day_wait': 500000.0, 'max_order_size': 40.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_JPY', 'currency_weights': {}, 'decay_frac': 0.5, 'target': 0.13203461479696113, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.08075876995235907, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 125.417388292345}], 'GBP_USD_sample_num': 1042676, 'GBP_CAD': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.007417924401730218, 'auc_barrier': 0.30843960866668196, 'min_trade_volatility': 57.497408951300685, 'max_pip_barrier': 176.07587115254432, 'day_wait': 500000.0, 'max_order_size': 181.64189643280616, 'reward_risk_ratio': 1.0522770620737438, 'currency_pair': 'GBP_CAD', 'currency_weights': {}, 'decay_frac': 0.24040543154158944, 'target': 0.05324918844156005, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.04213464619271697, 'is_max_volatility': True, 'adjust_factor': 2.499799200168823, 'max_barrier': 40.0, 'is_hedge': True, 'samples': 130.36542739116618}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': -0.0072429471833760145, 'auc_barrier': 0.9285862907603981, 'min_trade_volatility': 121.14584392781055, 'max_pip_barrier': 131.21926782301483, 'day_wait': 500000.0, 'max_order_size': 60.85240253878757, 'reward_risk_ratio': 3.0, 'currency_pair': 'GBP_CAD', 'currency_weights': {'JPY': 1.0, 'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'GBP': 0.3929125736212099, 'CAD': 1.0}, 'decay_frac': 0.1, 'target': 0.05996621864027644, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.04984310700589051, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 65.60499941870357, 'is_hedge': True, 'samples': 110.10698019277619}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.05030485773220197, 'target': 0.0708488164708447, 'min_trade_volatility': 10.0, 'max_pip_barrier': 249.97620815409167, 'day_wait': 500000.0, 'max_order_size': 159.2135563221226, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_CAD', 'currency_weights': {}, 'decay_frac': 0.2610281978961767, 'is_low_barrier': False, 'auc_barrier': 0.1, 'sharpe': 0.06767212163042818, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.47650652547543, 'max_barrier': 59.90507445424394, 'is_hedge': False, 'samples': 12.990149348585955}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.029425917224890616, 'target': 0.09413682430452157, 'min_trade_volatility': 60.45849001932747, 'max_pip_barrier': 249.98446228553186, 'day_wait': 500000.0, 'max_order_size': 141.0019819187659, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_CAD', 'currency_weights': {}, 'decay_frac': 0.18847236958875394, 'is_low_barrier': False, 'auc_barrier': 0.7148944357449188, 'sharpe': 0.06618310901965713, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 65.12767314572353, 'is_hedge': False, 'samples': 140.10803940260567}], 'EUR_CHF_sample_num': 973120, 'GBP_USD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.5369955864003223, 'mean_sharpe': 0.1179436225789731, 'target': 0.4348958496096444, 'max_pip_barrier': 202.50144505315225, 'day_wait': 500000.0, 'max_order_size': 60.00452729573725, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_USD', 'currency_weights': {}, 'min_trade_volatility': 124.71675907343, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.14759604720827021, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.4996688303021033, 'max_barrier': 65.71481391688909, 'is_hedge': True, 'samples': 213.80284775465498}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.3468174356462561, 'mean_sharpe': 0.04026312603135318, 'target': 0.04385136461681034, 'max_pip_barrier': 50.0, 'day_wait': 500000.0, 'max_order_size': 40.019485543006034, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_USD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 0.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 128.029142836722, 'is_low_barrier': False, 'auc_barrier': 1.6805157498753847, 'sharpe': 0.04083769405434434, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': True, 'samples': 97.70646221248631}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.6434891429574283, 'mean_sharpe': 0.1013834780473654, 'target': 0.15081480341634318, 'max_pip_barrier': 205.92896050196248, 'day_wait': 500000.0, 'max_order_size': 118.83356582052187, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_USD', 'currency_weights': {}, 'min_trade_volatility': 146.30440640908165, 'is_low_barrier': False, 'auc_barrier': 0.10017461002437446, 'sharpe': 0.13678522587754804, 'is_relavent_currency': False, 'is_max_volatility': True, 'adjust_factor': 2.498234848172309, 'max_barrier': 65.727042349099, 'is_hedge': False, 'samples': 10.234392113910186}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.4930777426117639, 'mean_sharpe': 0.12143691805344463, 'target': 0.20947580573018365, 'max_pip_barrier': 216.58985992795888, 'day_wait': 500000.0, 'max_order_size': 84.5897785209136, 'reward_risk_ratio': 1.0, 'currency_pair': 'GBP_USD', 'currency_weights': {'USD': 7.792451036626619e-10, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 139.90326858769515, 'is_low_barrier': False, 'auc_barrier': 0.44238910548786503, 'sharpe': 0.1738915859159471, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 22.227820372398686}], 'EUR_NZD_sample_num': 1391206, 'EUR_JPY': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.1999171384263999, 'auc_barrier': 2.0, 'min_trade_volatility': 49.92789508989803, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 81.44552638315281, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_JPY', 'currency_weights': {}, 'decay_frac': 0.9051832184015589, 'target': 4.954514453885574, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.22208938418100083, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 62.91587921173368, 'is_hedge': True, 'samples': 671.338914516772}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.15038979804183802, 'auc_barrier': 2.0, 'min_trade_volatility': 26.61682321038829, 'max_pip_barrier': 207.55647547569077, 'day_wait': 500000.0, 'max_order_size': 146.28393619125256, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_JPY', 'currency_weights': {}, 'decay_frac': 1.0, 'target': 1.9056731130944338, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.17316598582775555, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 45.44818572532408, 'is_hedge': True, 'samples': 458.64570875334584}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.11291634061564176, 'auc_barrier': 1.9138925806605, 'min_trade_volatility': 63.8271763134586, 'max_pip_barrier': 187.26627418511922, 'day_wait': 500000.0, 'max_order_size': 78.85432639884128, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_JPY', 'currency_weights': {}, 'decay_frac': 0.7530650070526891, 'target': 1.6450466206867629, 'is_low_barrier': False, 'is_relavent_currency': False, 'sharpe': 0.14534253554558504, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 590.4356740133264}, {'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.0634660860487696, 'target': 0.4340623268621074, 'min_trade_volatility': 64.34159145062426, 'max_pip_barrier': 249.9586061559409, 'day_wait': 500000.0, 'max_order_size': 122.73147081742385, 'reward_risk_ratio': 1.0, 'currency_pair': 'EUR_JPY', 'currency_weights': {}, 'decay_frac': 0.870143305744942, 'is_low_barrier': False, 'auc_barrier': 1.616457269393815, 'sharpe': 0.08293203017720349, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 100.0, 'is_hedge': False, 'samples': 364.77134233156784}], 'NZD_CHF_sample_num': 1525319, 'AUD_NZD_sample_num': 1377978, 'USD_JPY_sample_num': 1402497, 'NZD_CAD_sample_num': 1349628, 'CHF_JPY_sample_num': 966788, 'GBP_CHF_sample_num': 1235676, 'NZD_USD': [{'is_any_barrier': False, 'is_norm_signal': False, 'mean_sharpe': 0.14264708288956923, 'target': 0.17958580252933035, 'min_trade_volatility': 149.94852591335967, 'max_pip_barrier': 156.67007947527583, 'day_wait': 500000.0, 'max_order_size': 180.0, 'reward_risk_ratio': 2.9923126788017615, 'currency_pair': 'NZD_USD', 'currency_weights': {}, 'decay_frac': 0.1, 'is_low_barrier': False, 'auc_barrier': 1.930772525285927, 'sharpe': 0.16279804296996467, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 0.5222139928179568, 'max_barrier': 40.331899989434575, 'is_hedge': True, 'samples': 22.378305541519282}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.1589714877237507, 'auc_barrier': 2.0, 'min_trade_volatility': 126.6717746760509, 'max_pip_barrier': 140.83857858421442, 'day_wait': 500000.0, 'max_order_size': 167.3298387875451, 'reward_risk_ratio': 1.5, 'currency_pair': 'NZD_USD', 'currency_weights': {'JPY': 1.0, 'USD': 0.4270985173427755, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'GBP': 1.0, 'CAD': 1.0}, 'decay_frac': 0.1404848807147453, 'target': 0.25283368835058767, 'is_low_barrier': False, 'is_relavent_currency': True, 'sharpe': 0.19572915767920226, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 89.2024122124545, 'is_hedge': True, 'samples': 26.435910392454097}, {'is_any_barrier': False, 'is_norm_signal': True, 'mean_sharpe': 0.055820130279467826, 'target': 0.20531749560249327, 'min_trade_volatility': 132.6331520350044, 'max_pip_barrier': 170.4916285150423, 'day_wait': 500000.0, 'max_order_size': 100.0, 'reward_risk_ratio': 3.0, 'currency_pair': 'NZD_USD', 'currency_weights': {}, 'decay_frac': 0.6423370638822076, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.15804474670727345, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 30.60941841241566, 'is_hedge': False, 'samples': 33.32154286676772}, {'max_order_size': 158.14153524423057, 'currency_weights': {'JPY': 1.0, 'USD': 0.6206149177989907, 'AUD': 1.0, 'CHF': 1.0, 'NZD': 1.0, 'CAD': 1.0, 'GBP': 1.0, 'EUR': 1.0}, 'is_relavent_currency': True, 'max_barrier': 40.00550598428552, 'is_any_barrier': False, 'mean_sharpe': 0.02148330968535047, 'decay_frac': 0.5, 'samples': 19.919151086407275, 'sharpe': 0.18872985855653737, 'is_max_volatility': False, 'is_norm_signal': False, 'max_pip_barrier': 190.0878449260673, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_USD', 'auc_barrier': 1.4405756885288146, 'is_hedge': False, 'trade_volatility_factor': 0.1, 'target': 0.2132465105346043, 'day_wait': 500000.0, 'min_trade_volatility': 124.45351835212577, 'is_low_barrier': False, 'adjust_factor': 2.5}], 'USD_CHF': [{'max_order_size': 137.84541439022397, 'currency_weights': {}, 'is_relavent_currency': False, 'max_barrier': 65.01735702506966, 'is_any_barrier': False, 'mean_sharpe': 0.1132159879011092, 'decay_frac': 0.5, 'samples': 249.7850962880268, 'sharpe': 0.1686205685854101, 'is_max_volatility': False, 'is_norm_signal': False, 'max_pip_barrier': 200.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CHF', 'auc_barrier': 1.891024663635613, 'is_hedge': True, 'trade_volatility_factor': 0.1, 'target': 0.7260254789390923, 'day_wait': 500000.0, 'min_trade_volatility': 73.80453821420262, 'is_low_barrier': False, 'adjust_factor': 2.5}, {'is_any_barrier': False, 'is_norm_signal': True, 'decay_frac': 0.7851512614897533, 'mean_sharpe': 0.07046571513347884, 'target': 0.16913918783742138, 'max_pip_barrier': 142.33523269944143, 'day_wait': 500000.0, 'max_order_size': 200.0, 'reward_risk_ratio': 2.9599427988651827, 'currency_pair': 'USD_CHF', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 0.5840025208694889, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 50.0, 'is_low_barrier': False, 'auc_barrier': 0.7, 'sharpe': 0.10397893389715746, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.482437174203253, 'max_barrier': 40.00000000000001, 'is_hedge': True, 'samples': 63.56684342729557}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.7973122077683601, 'mean_sharpe': 0.1271525909307504, 'target': 0.5031437692388312, 'max_pip_barrier': 249.95241582548778, 'day_wait': 500000.0, 'max_order_size': 131.29730694973475, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CHF', 'currency_weights': {}, 'min_trade_volatility': 63.746727988196966, 'is_low_barrier': False, 'auc_barrier': 2.0, 'sharpe': 0.1413251124707052, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 52.87491138451343, 'is_hedge': False, 'samples': 344.32040189785096}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.5363068975986957, 'mean_sharpe': 0.0632652542388767, 'target': 0.1977198929199391, 'max_pip_barrier': 220.0, 'day_wait': 500000.0, 'max_order_size': 40.0, 'reward_risk_ratio': 1.0, 'currency_pair': 'USD_CHF', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 0.11445895011648266, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 1.0, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 26.5177435581151, 'is_low_barrier': False, 'auc_barrier': 1.0634496512309684, 'sharpe': 0.15857321313931327, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 1.4798358880506264, 'max_barrier': 40.0, 'is_hedge': False, 'samples': 72.93887803516607}], 'default_auc_mult': 1.371611543466972, 'NZD_USD_sample_num': 1570759, 'EUR_AUD_sample_num': 1465378, 'default_is_norm_signal': False, 'NZD_CAD': [{'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.23972420685011464, 'mean_sharpe': 0.1849759326048338, 'target': 0.5311047074805108, 'max_pip_barrier': 146.9348543321559, 'day_wait': 500000.0, 'max_order_size': 179.01621761686542, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_CAD', 'currency_weights': {}, 'min_trade_volatility': 115.90945426011585, 'is_low_barrier': False, 'auc_barrier': 1.5778326174679018, 'sharpe': 0.43363829384401187, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.499999997862696, 'max_barrier': 40.0, 'is_hedge': True, 'samples': 15.805504587155964}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 1.0, 'mean_sharpe': 0.01904503613612185, 'target': 0.07161245603242199, 'max_pip_barrier': 85.74215872298043, 'day_wait': 500000.0, 'max_order_size': 194.31667314154842, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_CAD', 'currency_weights': {}, 'min_trade_volatility': 50.0, 'is_low_barrier': False, 'auc_barrier': 0.6, 'sharpe': 0.06606808650155108, 'is_relavent_currency': True, 'is_max_volatility': True, 'adjust_factor': 2.5, 'max_barrier': 48.90439970044447, 'is_hedge': True, 'samples': 24.645871559633026}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.15604441812478678, 'mean_sharpe': 0.21386392382892672, 'target': 0.3839089883660503, 'max_pip_barrier': 167.80640080927037, 'day_wait': 500000.0, 'max_order_size': 158.10948844864458, 'reward_risk_ratio': 1.0, 'currency_pair': 'NZD_CAD', 'currency_weights': {}, 'min_trade_volatility': 89.24219048956756, 'is_low_barrier': False, 'auc_barrier': 1.0166287344951936, 'sharpe': 0.28159769472733703, 'is_relavent_currency': False, 'is_max_volatility': False, 'adjust_factor': 2.4996688303021033, 'max_barrier': 52.06691059882568, 'is_hedge': False, 'samples': 38.30825688073394}, {'is_any_barrier': False, 'is_norm_signal': False, 'decay_frac': 0.2445105653242359, 'mean_sharpe': 0.0901523650885714, 'target': 0.12690742151338877, 'max_pip_barrier': 200.0, 'day_wait': 500000.0, 'max_order_size': 180.0, 'reward_risk_ratio': 1.5, 'currency_pair': 'NZD_CAD', 'currency_weights': {'USD': 1.0, 'AUD': 1.0, 'CHF': 1.0, 'JPY': 1.0, 'GBP': 1.0, 'NZD': 0.145572812250556, 'EUR': 1.0, 'CAD': 1.0}, 'min_trade_volatility': 50.0, 'is_low_barrier': False, 'auc_barrier': 0.7, 'sharpe': 0.10952919707909328, 'is_relavent_currency': True, 'is_max_volatility': False, 'adjust_factor': 2.5, 'max_barrier': 90.86905365480207, 'is_hedge': False, 'samples': 26.25321100917431}]}
    else:
        adjust_factors = {'default_day_wait': {False: 10.029924216120842, True: 11.094367333486277}, 'EUR_NZD': [{'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'EUR_NZD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 15.0, 'is_low_barrier': False, 'sharpe': 1.407271131578543, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.7793449142228759, 'currency_pair': 'EUR_NZD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 4.684643263142805, 'is_low_barrier': False, 'sharpe': 1.1491993713272262, 'adjust_factor': 1.5094181640241158}, {'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'EUR_NZD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 15.0, 'is_low_barrier': False, 'sharpe': 0.3951037216068945, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5461692973843989, 'currency_pair': 'EUR_NZD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 5.235122536532051, 'is_low_barrier': False, 'sharpe': 1.0406033696351449, 'adjust_factor': 1.0494850143890921}], 'GBP_USD': [{'is_norm_signal': False, 'auc_barrier': 1.0, 'currency_pair': 'GBP_USD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 10.421493633218216, 'is_low_barrier': False, 'sharpe': 0.02192887986799922, 'adjust_factor': 2.5}, {'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'GBP_USD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 10.268887358033663, 'is_low_barrier': False, 'sharpe': 0.07839685589966673, 'adjust_factor': 0.8}, {'is_norm_signal': False, 'auc_barrier': 1.0, 'currency_pair': 'GBP_USD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 13.310660898844995, 'is_low_barrier': False, 'sharpe': 0.061306147973084396, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 1.0, 'currency_pair': 'GBP_USD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 3.0, 'is_low_barrier': False, 'sharpe': 0.053924673277123185, 'adjust_factor': 2.5}], 'EUR_CHF': [{'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'EUR_CHF', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 15.0, 'is_low_barrier': False, 'sharpe': 0.46434612981681134, 'adjust_factor': 0.8}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'EUR_CHF', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 7.144665106932409, 'is_low_barrier': False, 'sharpe': 0.38957384872927925, 'adjust_factor': 0.8}, {'is_norm_signal': False, 'auc_barrier': 0.5461692973843989, 'currency_pair': 'EUR_CHF', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 5.235122536532051, 'is_low_barrier': False, 'sharpe': 0.2801417519664906, 'adjust_factor': 1.0494850143890921}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'EUR_CHF', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 5.927768012846755, 'is_low_barrier': False, 'sharpe': 0.3341018264302853, 'adjust_factor': 2.5}], 'AUD_USD': [{'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'AUD_USD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 13.898926565237218, 'is_low_barrier': False, 'sharpe': -0.01196977774447409, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'AUD_USD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 3.0, 'is_low_barrier': False, 'sharpe': 0.03322079103160882, 'adjust_factor': 0.8}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'AUD_USD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 13.57312332638831, 'is_low_barrier': False, 'sharpe': 0.01211462951016326, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'AUD_USD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 13.123334527474794, 'is_low_barrier': False, 'sharpe': -0.010360377044772297, 'adjust_factor': 2.0553958702239674}], 'default_adjust': {False: 2.0688073790711807, True: 1.9435298688588654}, 'USD_CAD': [{'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'USD_CAD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 15.0, 'is_low_barrier': False, 'sharpe': 0.1452341789580169, 'adjust_factor': 2.5}, {'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'USD_CAD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 13.63901662935071, 'is_low_barrier': False, 'sharpe': 0.24476602719747984, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'USD_CAD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 15.0, 'is_low_barrier': False, 'sharpe': 0.11306612635974324, 'adjust_factor': 2.5}, {'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'USD_CAD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 13.412397538067985, 'is_low_barrier': False, 'sharpe': 0.16505810076426713, 'adjust_factor': 2.5}], 'GBP_NZD': [{'is_norm_signal': True, 'auc_barrier': 1.0, 'currency_pair': 'GBP_NZD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 15.0, 'is_low_barrier': False, 'sharpe': 0.24652999439760842, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'GBP_NZD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 12.909898310566547, 'is_low_barrier': False, 'sharpe': 1.5574118378734785, 'adjust_factor': 2.5}, {'is_norm_signal': True, 'auc_barrier': 1.0, 'currency_pair': 'GBP_NZD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 13.137438746252696, 'is_low_barrier': False, 'sharpe': 0.1217845869239829, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.860162246721079, 'currency_pair': 'GBP_NZD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 3.0013724978081386, 'is_low_barrier': False, 'sharpe': 0.746492791925786, 'adjust_factor': 1.5089374079943758}], 'default_auc_mult': 0.6332802055611697, 'default_is_norm_signal': False, 'NZD_CAD': [{'is_norm_signal': True, 'auc_barrier': 0.5, 'currency_pair': 'NZD_CAD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 9.738325029483882, 'is_low_barrier': False, 'sharpe': 0.21989618859417406, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 1.0, 'currency_pair': 'NZD_CAD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': True, 'is_any_barrier': False, 'day_wait': 9.615286772842412, 'is_low_barrier': False, 'sharpe': 0.005601646163313939, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'NZD_CAD', 'is_relavent_currency': False, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 14.057614762928772, 'is_low_barrier': False, 'sharpe': 0.2401862201294729, 'adjust_factor': 2.5}, {'is_norm_signal': False, 'auc_barrier': 0.5, 'currency_pair': 'NZD_CAD', 'is_relavent_currency': True, 'max_barrier': 100, 'is_hedge': False, 'is_any_barrier': False, 'day_wait': 7.4049836420152335, 'is_low_barrier': False, 'sharpe': 0.06490281788520225, 'adjust_factor': 0.8}]}

    return adjust_factors

def get_sharpe(is_low_barrier, is_hedge, select_pair, signals_file_prefix, model_type):

    adjust_factors = get_strategy_parameters(model_type)

    if select_pair not in adjust_factors:
        return None

    for item in adjust_factors[select_pair]:

        if item['is_low_barrier'] != is_low_barrier:
            continue

        if item['is_hedge'] != is_hedge:
            continue

        if 'is_relavent_currency' in item and item['is_relavent_currency'] != ("all" not in signals_file_prefix):
            continue

        if 'sharpe' in item:
            return item

        return None

    return None

def find_best_adjust_factor(is_low_barrier, is_hedge, select_pair, signals_file_prefix, is_max_barrier, model_type):

    adjust_factors = get_strategy_parameters(model_type)
    
    default_stop_adjust = adjust_factors["default_adjust"][is_hedge]
    default_auc_barrier_mult = adjust_factors["default_auc_mult"]
    default_is_norm_signal = adjust_factors["default_is_norm_signal"]

    if select_pair not in adjust_factors:
        if is_low_barrier:
            return 0.5, True, 100, default_auc_barrier_mult, True, 0.5, False, 1.0, 200, 130, 100, {}

        return default_stop_adjust, True, 100, default_auc_barrier_mult, True, 0.5, False, 1.0, 200, 130, 100, {}


    for item in adjust_factors[select_pair]:

        if item['is_low_barrier'] != is_low_barrier:
            continue

        if item['is_hedge'] != is_hedge:
            continue

        if 'is_relavent_currency' in item and item['is_relavent_currency'] != ("all" not in signals_file_prefix):
            continue

        max_barrier = 100
        is_max_volatility = False
        decay_frac = 0.5
        reward_risk_ratio = 1.0
        max_pip_barrier = 200
        max_order_size = 100
        min_trade_volatility = 130
        currency_weights = {}
        auc_barrier_mult = default_auc_barrier_mult
        is_normalize_signal = default_is_norm_signal

        if 'decay_frac' in item:
            decay_frac = item['decay_frac']

        if 'max_order_size' in item:
            max_order_size = item['max_order_size']

        if 'max_barrier' in item:
            max_barrier = item['max_barrier']

        if 'auc_barrier' in item:
            auc_barrier_mult = item['auc_barrier']

        if 'is_norm_signal' in item:
            is_normalize_signal = item['is_norm_signal']

        if 'is_max_volatility' in item:
            is_max_volatility = item['is_max_volatility']

        if 'reward_risk_ratio' in item:
            reward_risk_ratio = item['reward_risk_ratio']

        if 'max_pip_barrier' in item:
            max_pip_barrier = item['max_pip_barrier']

        if 'min_trade_volatility' in item:
            min_trade_volatility = item['min_trade_volatility']

        if 'currency_weights' in item:
            currency_weights = item['currency_weights']


        return item['adjust_factor'], item['is_any_barrier'], max_barrier, auc_barrier_mult, is_normalize_signal, decay_frac, is_max_volatility, reward_risk_ratio, max_pip_barrier, min_trade_volatility, max_order_size, {}

    return default_stop_adjust, True, 100, default_auc_barrier_mult, True, 0.5, False, 1.0, 200, 130, 100, {}

def close_all_orders_not_whitelistes(account_numbers, select_pair, white_list_order_ids):

    orders = []
    for account_number in account_numbers:
        orders1, total_margin = get_open_trades(account_number, {}, None, select_pair)
        orders += orders1

    for order in orders:
        if order.order_id not in white_list_order_ids:
            trade_logger.info('Orphan Order: ' + str(order.order_id))
            close_order(select_pair, order, None, None)

def process_pending_trades(account_numbers, avg_spreads, select_pair, signals_file_prefix, model_type, trading_params = None, is_exponential = False, strategy_weight = 1.0, is_low_barrier = False, is_max_barrier = False, is_new_trade = True, is_force_close = False, is_filter_member_orders = False, is_filter_members_hedge = False, is_recover = False):

    pip_size = 0.0001
    if select_pair[4:] == "JPY":
        pip_size = 0.01

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
    adjust_factor, is_any_barrier, max_barrier, auc_barrier_mult, is_normalize_signal, decay_frac, is_max_volatility, reward_risk_ratio, max_pip_barrier, min_trade_volatility, max_order_size, currency_weights = find_best_adjust_factor(is_low_barrier, is_hedge_account, select_pair, signals_file_prefix, is_max_barrier, model_type)

    if trading_params != None:
        decay_frac = 0.5
        if model_type == ModelType.time_regression:
            rmse_percentile = trading_params["rmse_percentile"]
            auc_barrier_mult = rmse_percentile
            if "is_normalize_signal" in trading_params:
                is_normalize_signal = trading_params["is_normalize_signal"]
            else:
                is_normalize_signal = 0
        else:
            auc_barrier_mult = trading_params["auc_barrier_mult"]

        min_trade_volatility = trading_params["min_trade_volatility"]

    else:
        rmse_percentile = "80th_percentile"

    growth_factor = ((total_balance + total_float_profit) / 5000) * strategy_weight
    if growth_factor == 0:
        is_new_trade = False
        growth_factor = ((total_balance + total_float_profit) / 5000)

    if os.path.isfile(root_dir + "group_metadata_news_release_" + select_pair + file_ext_key):
        with open(root_dir + "group_metadata_news_release_" + select_pair + file_ext_key, "rb") as f:
            try:
                group_metadata = pickle.load(f)
            except:
                group_metadata = {}
    else:
        group_metadata = {}

    if "pnls_times" not in group_metadata:
        group_metadata["pnls_times"] = []
        group_metadata["pnls"] = []

    order_metadata = {}

    if "max_equity" not in group_metadata:
        group_metadata["max_equity"] = total_balance

    if select_pair + "_last_signal_update_time" not in group_metadata:
        group_metadata[select_pair + "_last_signal_update_time"] = None

    group_metadata["max_equity"] = max(group_metadata["max_equity"], (total_balance + total_float_profit))

    if select_pair + "_orders" not in group_metadata:
        group_metadata[select_pair + "_orders"] = []

    orders = []
    total_margin = 0
    for account_number in account_numbers:
        orders1, total_margin = get_open_trades(account_number, order_metadata, total_margin, select_pair, is_force_close = is_force_close)
        if is_filter_member_orders:
            orders1 = [order for order in orders1 if order.order_id in group_metadata[select_pair + "_orders"]]
        orders += orders1

    new_order_ids = []
    if is_filter_member_orders:
        for order_id in group_metadata[select_pair + "_orders"]:
            found_order = False
            for order in orders:
                if order.order_id == order_id:
                    found_order = True
                    break

            if found_order:
                new_order_ids.append(order_id)

        group_metadata[select_pair + "_orders"] = new_order_ids

    with open(signals_file_prefix + select_pair + ".pickle", "rb") as f:
        try:
            base_calendar = pickle.load(f)
        except:
            return group_metadata[select_pair + "_orders"]

    if is_force_close:
        return group_metadata[select_pair + "_orders"]

    if len(orders) == 0 and is_new_trade == False:
        return group_metadata[select_pair + "_orders"]

    trade_volatility = float(base_calendar["month_std"]) / pip_size
    trade_logger.info('Equity: ' + str(total_balance + total_float_profit))

    if is_max_volatility:
        max_barrier = max(max_barrier, trade_volatility)

    orders_by_prediction = {}
    orders_by_model = {}
    last_order_time_diff_hours = {}
    existing_order_amount = set()

    float_profit = 0
    for order in orders:

        if order.prediction_key not in orders_by_prediction:
            orders_by_prediction[order.prediction_key] = []

        if order.model_key not in orders_by_model:
            last_order_time_diff_hours[order.model_key] = 9999999999
            orders_by_model[order.model_key] = []

        if order.pair + "_" + order.account_number + "_" + str(order.amount) in existing_order_amount:
            trade_logger.info("Duplicate FIFO trade")

        existing_order_amount.add(order.pair + "_" + order.account_number + "_" + str(order.amount))
        orders_by_prediction[order.prediction_key].append(order)
        orders_by_model[order.model_key].append(order)

        last_order_time_diff_hours[order.model_key] = min(last_order_time_diff_hours[order.model_key], order.time_diff_hours)

        float_profit += order.PnL

    if select_pair + "_growth_factor" not in group_metadata:
        group_metadata[select_pair + "_growth_factor"] = growth_factor

    if is_valid_trading_period(time.time()) and len(orders) > 0:

        group_metadata["pnls"].append(float_profit / group_metadata[select_pair + "_growth_factor"])
        group_metadata["pnls_times"].append(time.time())
        while len(group_metadata["pnls"]) > 1 and time.time() - group_metadata["pnls_times"][0] > 60 * 60 * 24 * 172:
            group_metadata["pnls"] = group_metadata["pnls"][1:]
            group_metadata["pnls_times"] = group_metadata["pnls_times"][1:]

    total_orders = len(orders)

    last_signal_update_time = time.ctime(os.path.getmtime(signals_file_prefix + select_pair + ".pickle"))
    print ("Last Modified", last_signal_update_time)

    orders = close_group_trades(select_pair, orders, False, \
        order_metadata, group_metadata, total_margin_available, total_margin_used, \
        existing_order_amount,  \
        growth_factor, is_exponential, adjust_factor, strategy_weight, 
        reward_risk_ratio, trade_volatility > min_trade_volatility)

    barrier_avg = 0
    barrier_count = 0
    shap_values = []
    description_forecast = {}

    if select_pair in base_calendar:
        for prediction in base_calendar[select_pair]:

            if is_exponential and prediction["barrier"] >= 60:
                continue

            if prediction["currency"] in currency_weights:
                currency_weight = currency_weights[prediction["currency"]]
            else:
                currency_weight = 1.0

            if currency_weight < 0.01:
                continue

            
            if model_type == ModelType.barrier:

                final_auc_mult = auc_barrier_mult * (trade_volatility / 100)

                if prediction["sharpe"] < 0 or prediction["barrier"] > max_barrier:
                    continue

                if prediction["auc"] < 0.51 and is_any_barrier == False:
                    continue

                if abs(prediction["probability"] - 0.5) < 0.5 - max(0, (prediction["auc"] - 0.5) * final_auc_mult):
                    continue

                if is_exponential:
                    barrier_avg += prediction["barrier"] * (prediction["probability"] - 0.5) * prediction["barrier"] * currency_weight
                else:
                    barrier_avg += prediction["barrier"] * (prediction["probability"] - 0.5) * currency_weight
                    
                barrier_count += abs(prediction["probability"] - 0.5) * currency_weight
            elif model_type == ModelType.time_regression:
                if abs(prediction["forecast"]) < prediction[rmse_percentile]:
                    continue

                if prediction["sharpe"] < 0:
                    continue

                if prediction["description"] not in description_forecast:
                    description_forecast[prediction["description"]] = []
                    norm = np.linalg.norm(prediction["shap_values"], ord=2) 
                    numerator = 1.0 / norm
                    shap_values.append([v * numerator for v in prediction["shap_values"]])

                barrier_count += 1
                description_forecast[prediction["description"]].append(prediction)

            elif model_type == ModelType.time_classification:
                if abs(prediction["probability"] - 0.5) < 0.5 - max(0, (prediction["auc"] - 0.5) * auc_barrier_mult):
                    continue

                barrier_avg += 50 * (prediction["probability"] - 0.5)
                barrier_count += abs(prediction["probability"] - 0.5) 
            else:
                sys.exit(-1)

    if model_type == ModelType.time_regression:
        for description in description_forecast:

            if is_normalize_signal == 0:
                num1 = sum([p["forecast"] / (p["rmse"]) for p in description_forecast[description]])
                den1 = sum([1.0 / p["rmse"] for p in description_forecast[description]])
                mean_rmse = np.mean([p["rmse"] for p in description_forecast[description]])

                barrier_avg += 25 * ((num1 / den1) * (1.0 / (mean_rmse)))

            elif is_normalize_signal == 2:
                barrier_avg += sum([p["forecast"] / (p["std_globals"]) for p in description_forecast[description]])



    if len(orders) > 0:
        print "time diff", last_order_time_diff_hours[select_pair], total_float_profit, float_profit
        trade_logger.info('PnL Barrier: ' + str(abs(np.mean(group_metadata["pnls"])) * 1.2))


    if is_normalize_signal == 1:
        signal = barrier_avg / max(1, barrier_count)
    else:
        signal = barrier_avg

    trade_logger.info('Signal: ' + str(signal)  + ",  Recent: " + str(barrier_avg)  + ",  AUC Mult: " + str(auc_barrier_mult)  + ",  is_norm_signal: " + str(is_normalize_signal)  + ",  barrier_count: " + str(barrier_count))
    trade_logger.info('Trade Volatility: ' + str(trade_volatility) + ",  Min Trade Volatility: " + str(min_trade_volatility) + ",  Decay Frac: " + str(decay_frac))

    if barrier_count > 0:

        if model_type == ModelType.time_regression:
            signal = min(signal, 200)
            signal = max(signal, -200)
        else:
            signal = min(signal, max_order_size)
            signal = max(signal, -max_order_size)
            
        if abs(signal) > 0:

            dir_prob = signal * ((900000 * 0.0001 * growth_factor)) * 0.5 
            if is_recover and "recover_factor" in group_metadata:
                dir_prob *= group_metadata["recover_factor"]

            if len(orders) > 0 and trade_volatility > min_trade_volatility:

                #key = "close_pnls" if (is_filter_member_orders and "close_pnls" in group_metadata) else "pnls"
                key = "pnls"
                new_update = (group_metadata[select_pair + "_last_signal_update_time"] != last_signal_update_time)
                is_force = new_update and ((float_profit / group_metadata[select_pair + "_growth_factor"]) > abs(np.mean(group_metadata[key])) * 1.2)

                orders = close_group_trades(select_pair, orders, is_force, \
                    order_metadata, group_metadata, total_margin_available, total_margin_used, \
                    existing_order_amount,  \
                    growth_factor, is_exponential, adjust_factor, strategy_weight, \
                    reward_risk_ratio, trade_volatility > min_trade_volatility, is_show_log = False)

            is_above_min_trade_volatility = trade_volatility > min_trade_volatility
            base_model_key = enter_group_trades(select_pair, orders, growth_factor, dir_prob, order_metadata, \
                group_metadata, orders_by_model, is_above_min_trade_volatility, \
                total_margin_available, total_margin_used, \
                account_numbers, existing_order_amount, \
                adjust_factor, strategy_weight,
                last_signal_update_time, reward_risk_ratio, max_pip_barrier, 
                decay_frac, model_type, shap_values, is_filter_member_orders, is_filter_members_hedge)

    with open(root_dir + "group_metadata_news_release_" + select_pair + file_ext_key, "wb") as f:
        pickle.dump(group_metadata, f)
    #pickle.dump(order_metadata, open(root_dir + "order_metadata_news_release_" + select_pair + file_ext_key, 'wb'))

    trade_logger.info('Total Orders: ' + str(total_orders)) 
    trade_logger.info('Max Value: ' + str(group_metadata["max_equity"])) 

    return group_metadata[select_pair + "_orders"]


def test_reduce_order():

    account_type = "fxpractice"
    account_number = "101-011-14579539-020"
    api_key = "09d62ba46dcb68fa1946dbacae79b90a-a684f7323963eba0093d2d13743c2600"
    curr_price = 1.08
    select_pair = "EUR_USD"

    '''
    order_info, account_number, order_amount = create_order("EUR_USD", curr_price, [account_number], True, 23, set(), "EUR_USD", curr_price - 0.01, curr_price + 0.01)

    print str(order_info)

    order_info = json.loads(order_info)

    if 'orderFillTransaction' in order_info:
        print "order successful"
    else:
        print "order fail"
    '''

    orders = []
    for account_number in [account_number]:
        orders1, total_margin = get_open_trades(account_number, {}, 0, select_pair, is_force_close = False)
        orders += orders1

    orders, order_amount = adjust_orders(select_pair, orders, 25, False)

    print ("final", order_amount)


def execute_straddle():
    pass

try:
    execute_straddle()
except:
    print (traceback.format_exc())

accounts = [
    ["101-011-9454699-002", "101-011-9454699-003"],
    ["101-011-9454699-004", "101-011-9454699-005"],
    ["101-011-9454699-006", "101-011-9454699-007"]
]

if get_mac() != 150538578859218:
    root_dir = "/root/" 
else:
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




