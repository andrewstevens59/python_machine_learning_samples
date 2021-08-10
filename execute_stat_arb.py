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
import os



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
		self.hedge_key = None
		self.margin_used = 0


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



def get_open_trades(account_number, order_metadata, total_margin):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades?count=50"

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))

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

			'''
			first_currency = pair[0:3]
			second_currency = pair[4:7]

			if first_currency != "USD":
				exposure_weight = avg_prices[first_currency + "_USD"]
			else:
				exposure_weight = 1

			if second_currency != "USD":
				pair_mult = avg_prices[second_currency + "_USD"]
			else:
				pair_mult = 1.0

			total_margin += float(margin_used)

			if amount > 0:
				price = avg_prices[pair + "_bid"]
			else:
				price = avg_prices[pair + "_ask"]

			if (amount > 0) == (avg_prices[pair] > open_price):
				print (abs(price - open_price) * abs(amount) * pair_mult), PnL, avg_prices[pair], pair_mult
			else:
				print (-abs(price - open_price) * abs(amount) * pair_mult), PnL, avg_prices[pair], pair_mult
			'''

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
			
			if key not in order_metadata:
				order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
				trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
				print "not exist", key
				continue

			metadata = order_metadata[key]

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.time_diff_hours = time_diff_hours
			order.order_id = order_id
			order.account_number = account_number
			order.model_key = metadata["model_key"]
			order.hedge_key = metadata["hedge_key"]
			order.dir = metadata["dir"]
			order.open_time = time_start
			order.open_price = open_price
			order.base_amount = metadata["base_amount"]
			order.sequence_id = metadata["sequence_id"]
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders, total_margin

def close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size):

	if (order.dir == True) == (curr_price > order.open_price):
		profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.base_amount
	else:
		profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.base_amount

	key = order.order_id + "_" + order.account_number
	if key in order_metadata:
		del order_metadata[key]

	existing_order_amount.remove(order.pair + "_" + order.account_number + "_" + str(order.amount))
	order_info, _ =  sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		trade_logger.info("Close Model Order: " + str({'model_key' : order.model_key, "Base Profit" : str(profit), "Actual Profit" : str(order.PnL)})) 
		return order_metadata, existing_order_amount, True

	return order_metadata, existing_order_amount, False

def simple_close_order(order, order_metadata, existing_order_amount):

	key = order.order_id + "_" + order.account_number

	existing_order_amount.remove(order.pair + "_" + order.account_number + "_" + str(order.amount))
	order_info, _ =  sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		trade_logger.info("Close Model Order: " + str({'model_key' : order.model_key, "Actual Profit" : str(order.PnL)})) 

		if key in order_metadata:
			del order_metadata[key]

		return order_metadata, existing_order_amount, True

	return order_metadata, existing_order_amount, False

def create_order(pair, curr_price, account_numbers, trade_dir, order_amount, existing_order_amount):

	if trade_dir == True:
		account_number = account_numbers[0]
	else:
		account_number = account_numbers[1]

	upper_inc = 0
	lower_inc = 0
	new_order_amount = order_amount
	fok_key = pair + "_" + account_number + "_" + str(order_amount)
	while fok_key in existing_order_amount:
		
		if rand.random() > 0.5 or lower_inc + 1 >= order_amount:
			upper_inc += 1
			new_order_amount = order_amount + upper_inc
			fok_key = pair + "_" + account_number + "_" + str(new_order_amount)
		else:
			lower_inc += 1
			new_order_amount = order_amount - lower_inc
			fok_key = pair + "_" + account_number + "_" + str(new_order_amount)

	existing_order_amount.add(fok_key)
	order_amount = new_order_amount

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
			"hedge_key" : order.hedge_key,
			"pair" : order.pair,
			"base_amount" : order.base_amount,
			"sequence_id" : order.sequence_id,
			"account_number" : order.account_number,
			"amount" : order.amount,
			"dir" : order.dir,
		}

	pickle.dump(group_metadata, open(root_dir + "group_metadata_other", 'wb'))
	pickle.dump(order_metadata, open(root_dir + "order_metadata_other", 'wb'))

def serialize_order(order):

	return {
			"model_key" : order.model_key,
			"hedge_key" : order.hedge_key,
			"base_amount" : order.base_amount,
			"pair" : order.pair,
			"open_time" : order.open_time,
			"open_price" : order.open_price,
			"order_id" : order.order_id,
			"sequence_id" : order.sequence_id,
			"account_number" : order.account_number,
			"amount" : order.amount,
			"dir" : order.dir,
		}


def close_group_trades(pair, portfolio_wts, orders_by_pair, order_metadata, group_metadata, total_margin_available, total_margin_used, existing_order_amount, curr_price, avg_prices, avg_spreads, total_models):

	if pair in orders_by_pair:
		orders = orders_by_pair[pair]
	else:
		orders = []

	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01

	commission = avg_spreads[pair] * pip_size

	new_orders = []
	for order in orders:

		margin_used_factor = (order.margin_used + -order.PnL) / (total_margin_available + total_margin_used)
		margin_used_factor /= abs(portfolio_wts[order.pair])

		trade_logger.info('Model: ' + order.model_key + \
			", Float Profit: " + str(order.PnL) + \
			", Dir: " + str(order.dir) + \
			", Portfolio Wt: " + str(portfolio_wts[order.pair]) + \
			", Margin Used: " + str(margin_used_factor))

		if (order.dir == True) == (curr_price > order.open_price):
			pip_diff = ((abs(curr_price - order.open_price)) / pip_size) 
		else:
			pip_diff = ((-abs(curr_price - order.open_price)) / pip_size)


		if margin_used_factor > 2.5: 
			order_metadata, existing_order_amount, is_success = close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size)
			if is_success == True:
				continue

		new_orders.append(order)

	if len(new_orders) != len(orders):
		update_orders(new_orders, group_metadata, order_metadata)


def enter_group_trades(pair, model_key, trade_dir, update_token, portfolio_wts, order_metadata, group_metadata, orders_by_pair, free_margin, used_margin, curr_price, account_numbers, existing_order_amount):

	if trade_dir == True:
		hedge_key = model_key + "_BUY" 
	else:
		hedge_key = model_key + "_SELL" 

	base_model_key = hedge_key + "_" + pair

	if pair in orders_by_pair:
		orders = orders_by_pair[pair]
	else:
		orders = []

	if base_model_key + "_update_token" in group_metadata:
		entry = (update_token != group_metadata[base_model_key + "_update_token"])
	else:
		entry = True

	if entry and free_margin > used_margin * 1.1:

		first_currency = pair[0:3]
		second_currency = pair[4:7]

		if second_currency != "USD":
			pair_mult = avg_prices[second_currency + "_USD"]
		else:
			pair_mult = 1.0

		new_order = Order()

		if trade_dir == False:
			new_order.base_amount = abs(portfolio_wts['wt'][pair]) * group_metadata[model_key + "_total_SELL"]
		else:
			new_order.base_amount = abs(portfolio_wts['wt'][pair]) * group_metadata[model_key + "_total_BUY"]

		trade_logger.info('Prev Model Wt: ' + hedge_key + ", : " + str(group_metadata[hedge_key + "_model_wt"])) 

		order_amount = int(round((group_metadata[hedge_key + "_model_wt"] * new_order.base_amount) / (pair_mult)))

		if order_amount == 0:
			return

		if (trade_dir) == (portfolio_wts['wt'][pair] > 0):
			new_order.dir = True
		else:
			new_order.dir = False

		order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount, existing_order_amount)

		print str(order_info)
		order_info = json.loads(order_info)

		if 'orderFillTransaction' in order_info:
			trade_logger.info('New Order: ' + str(order_info)) 
			order_id = str(order_info['orderFillTransaction']['id'])

			new_order.order_id = order_id
			new_order.amount = order_amount
			new_order.account_number = account_number
			new_order.model_key = model_key
			new_order.hedge_key = hedge_key
			new_order.open_time = time.time()
			new_order.open_price = curr_price
			new_order.pair = pair
			orders.append(new_order)

			trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 
			group_metadata[base_model_key + "_update_token"] = update_token

			update_orders(orders, group_metadata, order_metadata)

		else:
			trade_logger.info('Order Error: ' + str(order_info)) 

def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=D&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	prices = []
	times = []

	for index in range(len(j) - 1):
	    item = j[index]

	    times.append(item['time'])
	    prices.append(item['closeMid'])

	return prices, times


def calculate_stat_arb(portfolio_wts):

	returns = [0] * (252)
	for pair in portfolio_wts['wt']:

		prices, times = get_time_series(pair, 252)

		for i in range(len(prices) - 1):
			price_diff = prices[i + 1] - prices[i]
			returns[i] += portfolio_wts['wt'][pair] * price_diff

	price_series = [0]
	for ret in returns:
		price_series.append(price_series[-1] + ret)

	mean_price = np.mean(price_series)

	return price_series[-1] - mean_price, mean_price, price_series[-1] 

def process_pending_trades(account_numbers, avg_spreads, avg_prices):


	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	for account_number in account_numbers:
		response_value, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/summary", "GET")
		j = json.loads(response_value)

		account_profit = float(j['account'][u'unrealizedPL'])
		account_balance = float(j['account'][u'balance'])
		margin_available = float(j['account']['marginAvailable'])
		margin_used = float(j['account']['marginUsed'])

		total_balance += account_balance
		total_float_profit += account_profit
		total_margin_available += margin_available
		total_margin_used += margin_used

	trade_logger.info('Equity: ' + str(total_balance + total_float_profit))

	if os.path.isfile(root_dir + "portfolio_wts") == False:
		print "Can't find portfolio wts"
		trade_logger.info("Cannot Find Portfolio Wts")
		return 

	portfolio_wts = pickle.load(open(root_dir + "portfolio_wts", 'rb'))

	if os.path.isfile(root_dir + "group_metadata_other"):
		group_metadata = pickle.load(open(root_dir + "group_metadata_other", 'rb'))
	else:
		group_metadata = {}

	if os.path.isfile(root_dir + "order_metadata_other"):
		order_metadata = pickle.load(open(root_dir + "order_metadata_other", 'rb'))
	else:
		order_metadata = {}

	orders = []
	total_margin = 0
	for account_number in account_numbers:
		orders1, total_margin = get_open_trades(account_number, order_metadata, total_margin)
		orders += orders1

	orders_by_pair = {}
	orders_by_model = {}
	orders_by_hedge = {}
	existing_order_amount = set()

	for order in orders:

		if order.pair not in orders_by_pair:
			orders_by_pair[order.pair] = []

		if order.model_key not in orders_by_model:
			orders_by_model[order.model_key] = []

		if order.hedge_key not in orders_by_hedge:
			orders_by_hedge[order.hedge_key] = []

		existing_order_amount.add(order.pair + "_" + order.account_number + "_" + str(order.amount))
		orders_by_pair[order.pair].append(order)
		orders_by_model[order.model_key].append(order)
		orders_by_hedge[order.hedge_key].append(order)

	update_model_keys = set()
	for model_key in portfolio_wts:

		if model_key + "_last_update" not in group_metadata:
			group_metadata[model_key + "_last_update"] = 0
			group_metadata[model_key + "_total_BUY"] = 0
			group_metadata[model_key + "_total_SELL"] = 0

		if time.time() - group_metadata[model_key + "_last_update"] < 24 * 60 * 60:
			continue

		update_model_keys.add(model_key)
		grad_arb, mean_val, curr_price = calculate_stat_arb(portfolio_wts[model_key])
		total_orders = 0

		if grad_arb > 0:
			total_orders = group_metadata[model_key + "_total_SELL"]
		else:
			total_orders = group_metadata[model_key + "_total_BUY"]

		min_val = portfolio_wts[model_key]['min_val']
		max_val = portfolio_wts[model_key]['max_val']

		print curr_price, mean_val, min_val, max_val


		if grad_arb < 0:
			step_size = abs(min(curr_price, min_val) - mean_val) * 0.2
		else:
			step_size = abs(max(curr_price, max_val) - mean_val) * 0.2

		if abs(grad_arb) > step_size * total_orders:
			group_metadata[model_key + "_trade_dir"] = grad_arb < 0

			if grad_arb > 0:
				group_metadata[model_key + "_total_SELL"] += 1
			else:
				group_metadata[model_key + "_total_BUY"] += 1
		else:
			group_metadata[model_key + "_trade_dir"] = None

		if total_orders == 0:
			if group_metadata[model_key + "_trade_dir"] == True:
				group_metadata[model_key + "_BUY_model_wt"] = 1 * (total_balance / len(portfolio_wts))
			else:
				group_metadata[model_key + "_SELL_model_wt"] = 1 * (total_balance / len(portfolio_wts))

		group_metadata[model_key + "_update_token"] = rand.randint(0, 10e9)
		group_metadata[model_key + "_last_update"] = time.time()
		trade_logger.info("Portfolio: " + str({
			'model_key' : model_key,
			'dir' : group_metadata[model_key +"_trade_dir"],
			'total_orders' : total_orders,
			'time' : time.time(),
			'mean_price' : mean_val,
			'curr_price' : curr_price
			}))

		if model_key + "_BUY" in orders_by_hedge:
			trade_logger.info("Buy Trades: " + model_key + ", " + str(orders_by_hedge[model_key + "_BUY"]))

		if model_key + "_SELL" in orders_by_hedge:
			trade_logger.info("Sell Trades: " + model_key + ", " + str(orders_by_hedge[model_key + "_SELL"]))

		total_model_profit = 0
		if model_key in orders_by_model:
			for order in orders_by_model[model_key]:
				total_model_profit += order.PnL

		if total_model_profit > 0:

			trade_logger.info("Close All: " + model_key)
			for order in orders_by_model[model_key]:
				simple_close_order(order, order_metadata, existing_order_amount)

			group_metadata[model_key + "_total_BUY"] = 0
			group_metadata[model_key + "_total_SELL"] = 0
		else:

			total_buy_profit = 0
			if model_key + "_BUY" in orders_by_hedge:
				for order in orders_by_hedge[model_key + "_BUY"]:
					total_buy_profit += order.PnL

			if total_buy_profit > 0 and model_key + "_SELL" in orders_by_hedge:

				trade_logger.info("Close Sells: " + model_key)
				for order in orders_by_hedge[model_key + "_SELL"]:
					simple_close_order(order, order_metadata, existing_order_amount)

				group_metadata[model_key + "_total_SELL"] = 0

			total_sell_profit = 0
			if model_key + "_SELL" in orders_by_hedge:
				for order in orders_by_hedge[model_key + "_SELL"]:
					total_sell_profit += order.PnL

			if total_sell_profit > 0 and model_key + "_BUY" in orders_by_hedge:

				trade_logger.info("Close Buys: " + model_key)
				for order in orders_by_hedge[model_key + "_BUY"]:
					simple_close_order(order, order_metadata, existing_order_amount)

				group_metadata[model_key + "_total_BUY"] = 0

		pickle.dump(group_metadata, open(root_dir + "group_metadata_other", 'wb'))

	print group_metadata

	new_avg_price = {}
	not_found_last = False
	for pair in currency_pairs:
		response, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v1/prices?instruments=" + pair, "GET")
		response = json.loads(response)['prices']

		spread = avg_spreads[pair]

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

			if curr_spread < spread * 1.2:
				break

		new_avg_price[pair] = curr_price

		# Cycle through all the portfolios
		for model_key in portfolio_wts:

			if pair not in portfolio_wts[model_key]['wt']:
				# A portfolio doesn't necessarily contain every pair
				continue

			base_model_key = model_key + "_" + pair

			last_processed_key = base_model_key + "_last_processed"
			last_price_key = base_model_key + "_last_price"

			if last_price_key not in group_metadata:
				group_metadata[last_price_key] = curr_price
				not_found_last = True

			if last_processed_key not in group_metadata:
				group_metadata[last_processed_key] = time.time()
				not_found_last = True

			if model_key in update_model_keys or (time.time() - group_metadata[last_processed_key]) > 48 * 60 * 60:
				# The portfolio direction has just been updated
				group_metadata[last_processed_key] = time.time()
				group_metadata[last_price_key] = curr_price
				not_found_last = True

			price_diff = abs(curr_price - group_metadata[last_price_key]) / pip_size
			price_diff /= 10

			time_diff = abs(time.time() - group_metadata[last_processed_key]) 
			time_diff /= 60 * 60 * 2

			curr_spread /= max(max(1, price_diff), max(1, time_diff))
			
			if curr_spread > spread:
				print pair, curr_spread, spread, max(max(1, price_diff), max(1, time_diff)), time_diff
				continue

			'''
			close_group_trades(pair, portfolio_wts[model_key], orders_by_pair, order_metadata, group_metadata, \
				total_margin_available, total_margin_used, existing_order_amount, \
				curr_price, avg_prices, avg_spreads, len(currency_pairs))
			'''

			if group_metadata[model_key + "_trade_dir"] != None:

				enter_group_trades(pair, model_key, group_metadata[model_key + "_trade_dir"], \
					group_metadata[model_key + "_update_token"], portfolio_wts[model_key], \
					order_metadata, group_metadata, orders_by_pair, \
					total_margin_available, total_margin_used, curr_price, \
					account_numbers, existing_order_amount)

	if len(update_model_keys) > 0 or not_found_last == True:
		pickle.dump(group_metadata, open(root_dir + "group_metadata_other", 'wb'))

	for pair in currency_pairs:
		first_currency = pair[0:3]
		second_currency = pair[4:7]
		new_avg_price[second_currency + "_" + first_currency] = 1.0 / new_avg_price[pair]


	pickle.dump(new_avg_price, open(root_dir + "pair_avg_price", 'wb'))

	trade_logger.info('Total Orders: ' + str(len(orders))) 


accounts = [
	["101-011-9454699-002", "101-011-9454699-003"],
	["101-011-9454699-004", "101-011-9454699-005"]
]


if get_mac() != 154505288144005:
	avg_spreads = pickle.load(open("/root/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/root/pair_avg_price", 'rb'))
	root_dir = "/root/" 
else:
	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))
	root_dir = "/tmp/" 


	
trade_logger = setup_logger('first_logger', root_dir + "trade_other.log")

#process_pending_trades(["101-001-9145068-002", "101-001-9145068-003", "101-001-9145068-004", "101-001-9145068-005"]) #demo 2
process_pending_trades(accounts[1], avg_spreads, avg_prices) #demo
#process_pending_trades(["001-001-1370090-004", "001-001-1370090-003"])



if get_mac() == 154505288144005:

	print "Can't run locally - run only on remote server"

	root_dir = "/root/"
	
	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.put("/tmp/model_pair_wt", "/tmp/model_pair_wt")
	sftp.put("/tmp/pair_avg_spread", root_dir + "pair_avg_spread")
	sftp.put("/tmp/pair_avg_price", root_dir + "pair_avg_price")

	sftp.put("/tmp/pair_avg_spread", root_dir + "group_metadata_other")
	sftp.put("/tmp/pair_avg_price", root_dir + "order_metadata_other")

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))


	for pair in currency_pairs:
		sftp.put("/tmp/model_predictions_" + pair, "/tmp/model_predictions_" + pair)


