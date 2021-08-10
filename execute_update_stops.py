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
import datetime as dt
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
		self.prediction_key = None
		self.margin_used = 0
		self.open_prediction = 0
		self.curr_prediction = 0
		self.portfolio_wt = 0
		self.order_tag = None
		self.type = None
		
currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


def sendCurlRequest(url, request_type, post_data = None):
	response_buffer = StringIO()
	header_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, url)

	curl.setopt(pycurl.CUSTOMREQUEST, request_type)

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.setopt(curl.HEADERFUNCTION, header_buffer.write)


	if post_data != None:
		print post_data
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8', 'Content-Type: application/json'])
		curl.setopt(pycurl.POSTFIELDS, post_data)
	else:
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	header_value = header_buffer.getvalue()

	return response_value, header_value

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



def get_open_trades(account_number, order_tag):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/" + order_tag + "?count=50"

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
		open_orders = j[order_tag]

		for trade in open_orders:

			if 'trailingStopValue' in trade:
				continue

			order_id = trade['id']
			open_price = float(trade[u'price'])



			
			
			if 'currentUnits' in trade:
				amount = float(trade[u'currentUnits'])
				pair = trade[u'instrument']
				pair = trade['instrument'].replace("/", "_")
				trade_id = order_id
			else:
				amount = 0
				pair = None
				trade_id = trade[u'tradeID']


			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.order_id = order_id
			order.trade_id = trade_id
			order.account_number = account_number
			order.order_tag = order_tag

			orders.append(order)

	return orders


def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=America%2FNew_York")

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

		times.append(item['time'])
		prices.append(item['closeMid'])
		index += 1

	return prices, times



def process_pending_trades(account_numbers):

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	for account_number in account_numbers:
		response_value, _ = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/summary", "GET")
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

	orders = []
	for account_number in account_numbers:
		orders1 = get_open_trades(account_number, "trades")
		orders += orders1

	for account_number in account_numbers:
		orders1 = get_open_trades(account_number, "orders")
		orders += orders1

	orders_by_pair = {}
	orders_by_model = {}
	order_trade_id_exists = set()

	trade_id_map = {}

	for order in orders:

		if order.pair == None:
			order.pair = trade_id_map[order.trade_id].pair

		if order.pair not in orders_by_pair:
			orders_by_pair[order.pair] = []

		orders_by_pair[order.pair].append(order)

		if order.trade_id != None:
			trade_id_map[order.trade_id] = order

		if order.order_tag == "orders":
			order_trade_id_exists.add(order.trade_id)
			print "order_id", order.order_id, order.trade_id


	for pair in orders_by_pair:

		response, _ = sendCurlRequest("https://api-fxtrade.oanda.com/v1/prices?instruments=" + pair, "GET")
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

		precision = '%.4f'
		pip_val = 0.0001
		if pair[4:7] == 'JPY':
			precision = '%.2f'
			pip_val = 0.01


		for order in orders_by_pair[pair]:

			if order.dir:
				stop_loss = curr_price - (pip_val * 24)
			else:
				stop_loss = curr_price + (pip_val * 24)


			trade_logger.info("Update Stop: " + pair + ": " + str(stop_loss))

			if order.order_tag == "trades":
				if order.order_id not in order_trade_id_exists:
					order_info, _ =  sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + order.account_number + "/orders", "POST", '{"order" : { "type" : "STOP_LOSS", "tradeID" : "' + str(order.order_id) + '", "price" : "' + str(precision % stop_loss) + '"}}')
					print "U1", order.pair, order_info
			else:
				order_info, _ =  sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + order.account_number + "/orders/" + str(order.order_id), "PUT", '{"order" : { "type" : "STOP_LOSS", "tradeID" : "' + str(order.trade_id) + '", "price" : "' + str(precision % stop_loss) + '"}}')

				print "U2", order.order_id, order_info

	trade_logger.info("Finish")

accounts = [
	["001-001-1370090-003", "001-001-1370090-004"],
]


if get_mac() != 154505288144005:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 


	
trade_logger = setup_logger('first_logger', root_dir + "trade_stop_loss_update.log")

#process_pending_trades(["101-001-9145068-002", "101-001-9145068-003", "101-001-9145068-004", "101-001-9145068-005"]) #demo 2
process_pending_trades(accounts[0]) #demo
#process_pending_trades(["001-001-1370090-004", "001-001-1370090-003"])



