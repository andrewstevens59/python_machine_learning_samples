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
import json
import copy

import pickle
import math
import sys

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
import download_calendar as download_calendar
import string
import random as rand

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
import traceback
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
import os


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

api_key = "048d8db7aabba3b5766bd9f20c2d8bc8-4609f0a8906e29a9cce1006b1046d233"
file_ext_key = ""
account_type = "fxpractice"

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
import datetime as dt

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

def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(5000) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	rates = []
	prices = []
	labels = []
	times = []


	index = 0

	X = []
	y = []

	balance_map = {}
	while index < len(j):
		item = j[index]

		s = item['time']
		s = s[0 : s.index('.')]
		timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

		times.append(timestamp)
		prices.append(item['closeMid'])
		index += 1

	return prices, times

white_list = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

def get_open_trades(account_number, total_margin):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades?count=50"

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

		

			pip_size = 0.0001
			if pair[4:7] == "JPY":
				pip_size = 0.01

			week_day_start = datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").weekday()

			time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

			s = str(datetime.datetime.utcnow())
			s = s[0 : s.index('.')]

			week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

			time_end = time.time()

			time_diff_hours = time_end - time_start

			time_diff_hours /= 60 * 60

			week_num = int(time_diff_hours / (24 * 7))
			if week_num >= 1:
				time_diff_hours -= 48 * week_num

			if week_day_start != 6 and week_day_end < week_day_start:
				time_diff_hours -= 48
			elif week_day_end == 6 and week_day_end > week_day_start:
				time_diff_hours -= 48

			key = order_id + "_" + account_number
			
			'''
			if True:
				order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
				trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
				print "not exist", key
				continue
			'''

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.time_diff_hours = time_diff_hours
			order.order_id = order_id
			order.account_number = account_number
			order.open_time = trade['openTime']
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders, total_margin

def close_order(order):


	order_info, _ =  sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		return order_metadata, True

	return order_metadata, False

def create_order(pair, curr_price, account_numbers, trade_dir, order_amount):

	if trade_dir == True:
		account_number = account_numbers[0]
	else:
		account_number = account_numbers[1]


	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01

	precision = '%.4f'
	if pair[4:] == 'JPY':
		precision = '%.3f'

	if trade_dir == True:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (20000 * pip_size))) + '"}'
		#sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		#sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (20000 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		

	return order_info, account_number, order_amount

def update_orders(orders, group_metadata, order_metadata):

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
			"amount" : order.amount,
			"dir" : order.dir,
		}

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
			"amount" : order.amount,
			"dir" : order.dir,
		}


def close_group_trades(orders):


	total_pnl = 0

	for order in orders:
		total_pnl += order.PnL


	for order in orders:

		if total_pnl > 0: 
			order_metadata, is_success = close_order(order)
			if is_success == True:
				continue



def enter_group_trades(pair, orders, curr_price, avg_prices, curr_equity, reward_risk, account_numbers):


	first_currency = pair[0:3]
	second_currency = pair[4:7]

	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01

	if second_currency != "AUD":
		pair_mult = avg_prices[second_currency + "_AUD"]
	else:
		pair_mult = 1.0

	new_order = Order()
	new_order.base_amount = (curr_equity / 5000) * (len(orders) + 1) * 0.1

	if len(orders) > 0:
		new_order.dir = curr_price > orders[-1].open_price
		pip_diff = abs(curr_price - orders[-1].open_price) / pip_size
		trade_logger.info(pair + ' Pip Diff: ' + str(pip_diff)) 
		new_order.base_amount *= pip_diff 
	else:
		new_order.dir = reward_risk > 0
		new_order.base_amount *= 100

	if pair[4:7] == "JPY":
		order_amount = int(round((new_order.base_amount) / (100 * pair_mult)))
	else:
		order_amount = int(round((new_order.base_amount) / (pair_mult)))

	order_amount = max(1, order_amount)

	order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount)

	print str(order_info)
	order_info = json.loads(order_info)

	if 'orderFillTransaction' in order_info:
		trade_logger.info('New Order: ' + str(order_info)) 
		order_id = str(order_info['orderFillTransaction']['id'])

		new_order.order_id = order_id
		new_order.amount = order_amount
		new_order.account_number = account_number
		orders.append(new_order)

		trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 

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

def get_future_movement(a, b, time_offset, future_prices):


	future_time_periods = [1, 2, 3, 4, 5, 10, 15, 20]
	future_time_periods = [24 * t for t in future_time_periods]

	for period in future_time_periods:
		yfit = a + b * (time_offset + period)

		if period not in future_prices:
			future_prices[period] = []

		future_prices[period].append(yfit)


# solve for a and b
def best_fit(X, Y):

	b, a = np.polyfit(X, Y, 1)

	return a, b

def calculate_reward_risk(pair, pip_size):

	prices, times = get_time_series(pair, 365, granularity="D")

	min_prices = min(prices)
	max_prices = max(prices)
	pip_range = (max(prices) - min(prices)) 
	pip_range /= 2

	X_set = []
	y_set = []

	future_prices = {}
	future_time_periods = [1, 2, 3, 4, 5, 10, 15, 20]
	for look_back in range(20, len(prices), 20):
		Y = prices[len(prices)-look_back:]
		X = [len(prices) - look_back + x for x in range(len(Y))]

		# solution
		a, b = best_fit(X, Y)

		yfit = [a + b * xi for xi in X]


		if abs(yfit[-1] - Y[-1]) < pip_range:
			X_set.append(X)
			y_set.append(yfit)

			get_future_movement(a, b, len(prices), future_prices)

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

			yfit = [a + b * xi for xi in X]

			if abs(yfit[-1] - Y[-1]) < pip_range:
				X_set.append(X)
				y_set.append(yfit)
				get_future_movement(a, b, len(prices), future_prices)


	dists = []
	price_levels = []

	for time_frame, future_period in zip(["1 Day", "2 Days", "3 Days", "4 Days", "1 Week", "2 Weeks", "3 Weeks", "1 Month"], [1, 2, 3, 4, 5, 10, 15, 20]):
		future_period *= 24
		dists += [(y - prices[-1]) / pip_size for y in future_prices[future_period]]
		price_levels += future_prices[future_period]


	left_price_levels = []
	right_price_levels = []
	reward_risk_ratio_map = {}
	reward_risk_rations = []

	for percenitle in range(5, 55):
		supports = [v for v in price_levels if v < prices[-1]]
		left_line = np.percentile(supports, 100 - percenitle)
		left_price_levels.append(left_line)

		resistances = [v for v in price_levels if v > prices[-1]]
		right_line = np.percentile(resistances, percenitle)
		right_price_levels.append(right_line)

		if abs(left_line - prices[-1]) > abs(right_line - prices[-1]):
			reward_risk_rations.append(-abs(left_line - prices[-1])  / abs(right_line - prices[-1]))
		else:
			reward_risk_rations.append(abs(right_line - prices[-1])  / abs(left_line - prices[-1]))

		reward_risk_ratio_map[percenitle] = reward_risk_rations[-1]

	return np.mean(reward_risk_rations)

def process_pending_trades(account_numbers, select_pairs):

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	for account_number in account_numbers:
		response_value, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/summary", "GET")
		j = json.loads(response_value)
		print(j)

		account_profit = float(j['account'][u'unrealizedPL'])
		account_balance = float(j['account'][u'balance'])
		margin_available = float(j['account']['marginAvailable'])
		margin_used = float(j['account']['marginUsed'])

		total_balance += account_balance
		total_float_profit += account_profit
		total_margin_available += margin_available
		total_margin_used += margin_used

	trade_logger.info('Equity: ' + str(total_balance + total_float_profit))


	avg_prices = {}
	pair_bid_ask_map = {}
	for pair in currency_pairs:
		response, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v1/prices?instruments=" + pair, "GET")
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

		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[first_currency + "_" + second_currency] = curr_price

	for pair in currency_pairs:
		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[second_currency + "_" + first_currency] = 1.0 / avg_prices[pair]


	orders = []
	total_margin = 0
	for account_number in account_numbers:
		orders1, total_margin = get_open_trades(account_number, total_margin)
		orders += orders1

	for pair in select_pairs:

		select_orders = [order for order in orders if order.pair == pair]
		select_orders = sorted(select_orders, key=lambda order: order.open_time)
		total_pnl = sum([order.PnL for order in orders]) 

		close_group_trades(select_orders)

		pip_size = 0.0001
		if pair[4:] == "JPY":
			pip_size = 0.01

		curr_spread = (pair_bid_ask_map[pair]['bid'] - pair_bid_ask_map[pair]['ask']) / pip_size
		if curr_spread > 5:
			trade_logger.info(pair + ' Spread To Big') 
			continue

		reward_risk = calculate_reward_risk(pair, pip_size)
		trade_logger.info(pair + ' Orders: ' + str(len(select_orders))) 
		trade_logger.info(pair + ' PnL: ' + str(total_pnl)) 
		trade_logger.info(pair + ' Reward Risk: ' + str(reward_risk)) 

		if abs(reward_risk) > 2:
			enter_group_trades(pair, select_orders, avg_prices[pair], avg_prices, total_balance + total_float_profit, reward_risk, account_numbers)


	trade_logger.info('Global PnL: ' + str(total_balance + total_float_profit) + ", Global Margin: " + str(total_margin_used * 0.02))
	trade_logger.info('Total Orders: ' + str(len(orders)))


accounts = [
	["101-011-9454699-002", "101-011-9454699-003"],
	["101-011-14392464-002", "101-011-14392464-002"],
	["101-011-14392464-002", "101-011-14392464-002"],
]


if get_mac() != 150538578859218:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 

trade_logger = setup_logger('first_logger', root_dir + "trade_flipper.log")


try:
    process_pending_trades(accounts[2], ["AUD_USD", "EUR_USD", "USD_JPY", "GBP_USD"]) 
    trade_logger.info('Finished ') 
except:
    print (traceback.format_exc())
    trade_logger.info(traceback.format_exc())


