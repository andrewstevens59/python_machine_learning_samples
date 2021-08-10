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
import time
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


import mysql.connector

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

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

if get_mac() != 154505288144005:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 

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

trade_logger = setup_logger('first_logger', root_dir + "trade_setup.log")


def sendCurlRequest(api_key, url, request_type, post_data = None):
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

def create_order(api_key, pair, curr_price, account_number, trade_dir, order_amount, open_price, barrier_size, base_dir):

	if open_price < 0:
		open_price = curr_price

	if base_dir:
		stop_loss_price = open_price - barrier_size
		take_profit_price = open_price + barrier_size
	else:
		stop_loss_price = open_price + barrier_size
		take_profit_price = open_price - barrier_size

	if trade_dir != base_dir:
		temp = stop_loss_price
		stop_loss_price = take_profit_price
		take_profit_price = temp

	print "order", stop_loss_price, take_profit_price, open_price, barrier_size

	precision = '%.4f'
	if pair[4:] == 'JPY':
		precision = '%.2f'

	if trade_dir == True:
		tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % take_profit_price) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
		order_info, _ = sendCurlRequest(api_key, "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ',' + tp_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % take_profit_price) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
		order_info, _ = sendCurlRequest(api_key, "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ',' + tp_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		

	return order_info, account_number, order_amount

def get_open_trades(api_key, account_number):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades?count=50"

	while next_link != None:
		response_value, header_value = sendCurlRequest(api_key, next_link, "GET")

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

			print trade


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

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.time_diff_hours = time_diff_hours
			order.order_id = order_id
			order.account_number = account_number
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders

def get_open_trades(api_key, account_number, order_tag):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/" + order_tag + "?count=50"

	while next_link != None:
		response_value, header_value = sendCurlRequest(api_key, next_link, "GET")

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
			order_id = trade['id']
			open_price = float(trade[u'price'])
			
			
			if 'currentUnits' in trade:
				amount = float(trade[u'currentUnits'])
				pair = trade[u'instrument']
				pair = trade['instrument'].replace("/", "_")
				PnL = float(trade['unrealizedPL'])
				margin_used = float(trade['marginUsed'])

				trade_id = order_id
			else:
				amount = 0
				pair = None
				PnL = None
				margin_used = None
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
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders

def close_order(order, api_key):

	order_info, _ =  sendCurlRequest(api_key, "https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		return True

	return False

def get_price(pair):
	response, _ = sendCurlRequest(api_key, "https://api-fxpractice.oanda.com/v1/prices?instruments=" + pair, "GET")
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

	return curr_price

cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='66.33.203.187',
                              database='newscapital')
cursor = cnx.cursor()

query = ("SELECT * FROM trade_setups")


cursor.execute(query)

setup_rows = []
for row in cursor:
	setup_rows.append(row)

cursor.close()

closed_setup = set()

order_map = {}
stop_loss_map = {}
for row in setup_rows:
	api_key = row[0]
	pair = row[1]
	stop_loss = row[2]
	open_price = row[3]
	max_positions = row[4]
	max_exposure = row[5]
	sizing_factor = row[6]
	spacing_factor = row[7]
	trade_dir = row[8]
	take_profit = row[9]
	account_number = row[10]

	total_pnl = row[11]
	total_orders = row[12]
	total_margin_used = row[13]
	last_update_time = row[14]
	min_profit = row[15]
	barrier_in_pips = row[16]


	if api_key + "_" + pair in closed_setup:
		continue

	print row

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0

	response_value, _ = sendCurlRequest(api_key, "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/summary", "GET")
	j = json.loads(response_value)

	account_profit = float(j['account'][u'unrealizedPL'])
	account_balance = float(j['account'][u'balance'])
	margin_available = float(j['account']['marginAvailable'])
	margin_used = float(j['account']['marginUsed'])

	total_balance += account_balance
	total_float_profit += account_profit
	total_margin_available += margin_available
	total_margin_used += margin_used

	if api_key + account_number not in order_map:
		orders = get_open_trades(api_key, account_number, "trades")
		order_map[api_key + account_number] = orders

		trade_id_map = {}
		for order in orders:
			 trade_id_map[order.trade_id] = order

		pending = get_open_trades(api_key, account_number, "orders")

		print "------------------------"

		for order in pending:
			trade_id_map[order.trade_id].stop_loss = order.open_price
			stop_loss_map[trade_id_map[order.trade_id].pair + "_" + api_key] = order.open_price

			print trade_id_map[order.trade_id].pair + "_" + api_key
	else:
		orders = order_map[api_key + account_number]

	orders = [order for order in orders if order.pair == pair]

	total_order_pnl = 0
	total_order_margin = 0
	for order in orders:
		total_order_pnl += order.PnL
		total_order_margin += order.margin_used


	min_profit = min(min_profit, total_order_pnl)
	

	if len(orders) > 0:
		cursor = cnx.cursor()
		query = ("UPDATE trade_setups SET min_profit = '" + str(min_profit) + "', stop_loss = '" + str(stop_loss_map[pair + "_" + api_key]) + "', last_update_time = '" + str(time.time()) + "', total_pnl = '" + str(total_order_pnl) + "', total_orders = '" + str(len(orders)) + "', total_margin_used = '" + str(total_order_margin) + "'  where api_key='" + api_key + "' and pair='" + pair + "'")
		cursor.execute(query)
		cnx.commit()
		cursor.close()

	if pair[4:7] != "AUD":
		if pair[4:7] + "_AUD" in currency_pairs:
			base_price = get_price(pair[4:7] + "_AUD")
		else:
			base_price = 1.0 / get_price("AUD_" + pair[4:7])
	else:
		base_price = 1.0
	
	curr_price = get_price(pair)

	if (len(orders) == 0) and (open_price > 0):

		if abs(time.time() - last_update_time) > 5 * 60 and last_update_time > 0:
			cursor = cnx.cursor()
			query = ("DELETE FROM trade_setups where api_key='" + api_key + "' and pair= '" + pair + "'")
			cursor.execute(query)
			cnx.commit()
			cursor.close()

			closed_setup.add(api_key + "_" + pair)
			trade_logger.info('Remove From DB: ' + api_key + " " + pair) 
		continue

	if pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	barrier_size = barrier_in_pips * pip_size

	if trade_dir:
		stop_loss = open_price - barrier_size
	else:
		stop_loss = open_price + barrier_size

	total_amount = 0
	for i in range(max_positions - 1):
		if spacing_factor < 1.0:
			curr_barrier_size *= spacing_factor
		else:
			curr_barrier_size = barrier_size * (float(max_positions - i) / (max_positions))

		total_amount += 1.0 / curr_barrier_size

	curr_barrier_size = barrier_size
	for i in range(len(orders)):
		if spacing_factor < 1.0:
			curr_barrier_size *= spacing_factor
			order_amount = 1.0 / curr_barrier_size
		else:
			curr_barrier_size = barrier_size * (float(max_positions - i) / (max_positions))
			order_amount = 1.0 / curr_barrier_size

			curr_barrier_size = barrier_size * (float(max_positions - i - 2) / (max_positions - 1))

		if trade_dir:
			next_level = stop_loss + curr_barrier_size
		else: 
			next_level = stop_loss - curr_barrier_size

	if len(orders) == 0:
		order_amount = round(total_balance * max_exposure * 0.01 * 100)
	else:
		order_amount = round((float(order_amount) / total_amount) * total_balance * max_exposure * 0.01 * 100)

	order_amount *= 60.0 / barrier_in_pips

	if len(orders) >= max_positions:
 		continue

	if (len(orders) == 0) or ((curr_price < next_level) == trade_dir):

		if pair[4:7] == "JPY":
			order_amount /= base_price * 100
		else:
			order_amount /= base_price

		order_amount = round(order_amount)

		if len(orders) == 0:
			hedge_trade_dir = trade_dir
		else:
			hedge_trade_dir = (trade_dir == False)

		order_info, account_number, order_amount = create_order(api_key, pair, curr_price, account_number, hedge_trade_dir, order_amount, open_price, barrier_size, trade_dir)

		print str(order_info)

		if 'orderFillTransaction' in order_info:

			if len(orders) == 0 or open_price < 0:
				cursor = cnx.cursor()
				query = ("UPDATE trade_setups SET open_price = '" + str(curr_price) + "' where api_key='" + api_key + "' and pair='" + pair + "'")
				cursor.execute(query)
				cnx.commit()
				cursor.close()

				trade_logger.info('Open First Trade: ' + api_key + " " + pair) 

			cursor = cnx.cursor()
			query = ("UPDATE trade_setups SET total_pnl = '" + str(total_order_pnl) + "', total_orders = '" + str(len(orders) + 1) + "', total_margin_used = '" + str(total_order_margin) + "'  where api_key='" + api_key + "' and pair='" + pair + "'")
			cursor.execute(query)
			cnx.commit()
			cursor.close()

			trade_logger.info('Update Trade Setups: ' + api_key + " " + pair) 
		elif len(orders) == 0 or open_price < 0:
			cursor = cnx.cursor()
			query = ("DELETE FROM trade_setups where api_key='" + api_key + "' and pair= '" + pair + "'")
			cursor.execute(query)
			cnx.commit()
			cursor.close()

			closed_setup.add(api_key + "_" + pair)
			trade_logger.info('Remove From DB: ' + api_key + " " + pair) 


cnx.close()



