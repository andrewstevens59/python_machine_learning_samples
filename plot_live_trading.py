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

			if amount > 0:
				price = bid
			else:
				price = ask

			if (amount > 0) == (avg_prices[pair] > open_price):
				print (abs(price - open_price) * abs(amount) * pair_mult), PnL, avg_prices[pair], pair_mult, "***"
			else:
				print (-abs(price - open_price) * abs(amount) * pair_mult), PnL, avg_prices[pair], pair_mult, "***"
			

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

		s = item['time'].replace(':', "-")
		s = s[0 : s.index('.')]
		time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

		times.append(time_start)
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

	return price_series, times

def find_closest_price(price_series, times, time_find):

	min_diff = 99999999
	best_price = 0
	for index in range(len(times)):

		price = price_series[index]
		time = times[index]


		if abs(time_find - time) < min_diff:
			min_diff = abs(time_find - time)
			best_price = price

	return best_price


def process_pending_trades(account_numbers, avg_spreads, avg_prices):

	if os.path.isfile("/tmp/portfolio_wts") == False:
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

	import matplotlib.pyplot as plt

	print avg_prices

	for model_key in portfolio_wts:
		
		net_price_mov = 0
		total_pnl = 0
		total_expected_pnl = 0
		min_open_time = 999999999999999
		if model_key in orders_by_model:

			price_series, times = calculate_stat_arb(portfolio_wts[model_key])

			y_offset = 0
			for order in orders_by_model[model_key]:

				min_open_time = min(min_open_time, order.open_time)
				response, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v1/prices?instruments=" + order.pair, "GET")
				response = json.loads(response)['prices']

				pip_size = 0.0001
				if order.pair[4:] == "JPY":
					pip_size = 0.01

				bid = None
				ask = None
				for spread_count in range(1):
					curr_spread = 0
					for price in response:
						if price['instrument'] == order.pair:
							curr_price = (price['bid'] + price['ask']) / 2
							curr_spread = abs(price['bid'] - price['ask']) / pip_size
							bid = price['bid']
							ask = price['ask']
							break

				if order.dir == True:
					curr_price = bid
				else:
					curr_price = ask

				net_price_mov += (curr_price - order.open_price) * portfolio_wts[model_key]['wt'][order.pair]


				first_currency = order.pair[0:3]
				second_currency = order.pair[4:7]

				if second_currency != "USD":
					pair_mult = avg_prices[second_currency + "_USD"]
				else:
					pair_mult = 1.0

				if order.dir == True:
					expected_pnl = abs(order.amount) * (curr_price - order.open_price) * pair_mult
				else:
					expected_pnl = -abs(order.amount) * (curr_price - order.open_price) * pair_mult

				total_pnl += order.PnL
				total_expected_pnl += expected_pnl

				print order.pair, "actual", order.PnL, "expected", expected_pnl, "amount", abs(order.amount), "open", order.open_price, "current", curr_price, "mult", pair_mult

				#print order.pair, "total actual", total_pnl, "total expected",  total_expected_pnl
				plt.plot(order.open_time, find_closest_price(price_series, times, order.open_time) - y_offset, 'ro')

				text = order.pair + ": " + str(total_pnl)

				if order.dir == True:
					text += " B"
				else:
					text += " S"
				plt.annotate(text, (order.open_time - 6000000, find_closest_price(price_series, times, order.open_time)  - y_offset))

				y_offset += 0.01

		'''
		if group_metadata[model_key + "_trade_dir"] == True:
			plt.plot([min_open_time, time.time()],[0, 10000 * net_price_mov])
			plt.annotate("Projected " + str(total_pnl), (time.time() - 100, 10000 * net_price_mov))
		else:
			plt.plot([min_open_time, time.time()],[0, -10000 * net_price_mov])
			plt.annotate("Projected " + str(total_pnl), (time.time() - 100, -10000 * net_price_mov))
		'''

		if abs(total_expected_pnl) > 0:

			plt.plot(times, price_series[:len(times)])
			plt.title(str(portfolio_wts[model_key]['wt']))
			plt.xlabel("IsBuy " + str(group_metadata[model_key + "_trade_dir"]) + " total " + str(total_pnl))
			plt.show()

			print order.pair, "total actual", total_pnl, "total expected", total_expected_pnl

			print ""


accounts = [
	["101-011-9454699-002", "101-011-9454699-003"],
	["101-011-9454699-004", "101-011-9454699-005"]
]


avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))
root_dir = "/tmp/" 

trade_logger = setup_logger('first_logger', root_dir + "trade_other.log")


if get_mac() == 154505288144005:

	print "Can't run locally - run only on remote server"
	
	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.get("/root/group_metadata_other", "/tmp/group_metadata_other")
	sftp.get("/root/order_metadata_other", "/tmp/order_metadata_other")
	sftp.get("/root/pair_avg_price", "/tmp/pair_avg_price")

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))


#process_pending_trades(["101-001-9145068-002", "101-001-9145068-003", "101-001-9145068-004", "101-001-9145068-005"]) #demo 2
process_pending_trades(accounts[1], avg_spreads, avg_prices) #demo
#process_pending_trades(["001-001-1370090-004", "001-001-1370090-003"])



