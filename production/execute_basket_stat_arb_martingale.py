import sys
import math
from datetime import datetime
from random import *
import os.path


import pickle
from scipy import stats
from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from datetime import timedelta
import mysql.connector
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
import string
import random as rand
import pycurl

import os
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from uuid import getnode as get_mac
import traceback
import socket
import paramiko
import json

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

params = {'USD': {'sl_pips': 90, 'tp_pips': 50}, 'AUD': {'sl_pips': 70, 'tp_pips': 50}, 'CHF': {'sl_pips': 50, 'tp_pips': 110}, 'JPY': {'sl_pips': 60, 'tp_pips': 70}, 'GBP': {'sl_pips': 50, 'tp_pips': 70}, 'NZD': {'sl_pips': 90, 'tp_pips': 50}, 'EUR': {'sl_pips': 60, 'tp_pips': 70}, 'CAD': {'sl_pips': 50, 'tp_pips': 50}}

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

	print (url)
	print (api_key)


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
	print (response_value)

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
	next_link = "https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades?count=50"

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
			order.order_id = int(order_id)
			order.account_number = account_number
			order.open_time = time_start
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders, total_margin

def close_order(account_number, order_id):

	if order_id == -1:
		return True


	order_info, _ =  sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades/" + str(order_id) + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		return True

	return False

def create_order(pair, curr_price, account_numbers, trade_dir, order_amount, stop_loss):


	account_number = account_numbers[0]


	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01

	precision = '%.4f'
	if pair[4:] == 'JPY':
		precision = '%.3f'

	if trade_dir == True:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (20000 * pip_size))) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (stop_loss * pip_size))) + '"}'
		#gsl_price = '"guaranteedStopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (170 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
		print ("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (stop_loss * pip_size))) + '"}'
		#gsl_price = '"guaranteedStopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price + (170 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		

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



def enter_group_trades(select_currency, pair, curr_price, avg_prices, account_numbers, is_buy, max_equity, order_amount, open_delta, revert_count):


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
	new_order.dir = is_buy

	if order_amount is None:
		new_order.base_amount = (max_equity / 5000) * 4000 * abs(open_delta)
	else:
		new_order.base_amount = order_amount


	if pair[4:7] == "JPY":
		order_amount = int(round((new_order.base_amount) / (100 * pair_mult)))
	else:
		order_amount = int(round((new_order.base_amount) / (pair_mult)))

	order_amount = max(1, order_amount)

	#stop_loss = int(round(max(25, params[select_currency]["sl_pips"] / (abs(open_delta) * revert_count))))
	stop_loss = 50

	order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount, stop_loss)

	print str(order_info)
	order_info = json.loads(order_info)

	if 'orderFillTransaction' in order_info:
		trade_logger.info('New Order: ' + str(order_info)) 
		order_id = int(order_info['orderFillTransaction']['id'])

		new_order.order_id = order_id
		new_order.amount = order_amount
		new_order.account_number = account_number

		trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 

	else:
		order_id = None
		trade_logger.info('Order Error: ' + str(order_info)) 

	return order_id, new_order.base_amount

def get_time_series(symbol, time, granularity="H1"):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=" + granularity + "&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 8bde4f67a710b42553a821bdfff8efa9-eb1cb834f4060df9949504beb3356265'])

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

def open_reverse_trade(pair_order_id, pair, is_revert, revert_count, order_amount, account_number, select_currency, avg_prices, is_buy, open_delta, position_id):

	print ("opening order", pair, is_buy)
	order_id, base_amount = enter_group_trades(select_currency, pair, avg_prices[pair], avg_prices, [account_number], is_buy, None, order_amount, open_delta, revert_count)

	if order_id is not None:

		cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

		cursor = cnx.cursor()

		query = ("""INSERT INTO stat_arb_single(user_id, pair, is_buy, open_timestamp, base_currency, pair_order_id, account_nbr, open_delta, is_revert, revert_count, order_amount)
			values ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')""".
			format(
				1,
				pair,
				1 if is_buy else 0,
				time.time(),
				select_currency,
				order_id,
				account_number,
				open_delta, 
				1 if is_revert else 0,
				revert_count,
				order_amount
				))

		print (query)

		cursor.execute(query)
		cnx.commit()


		query = ("""update stat_arb_single set is_close=1 
					where 
					account_nbr='{}' and base_currency='{}' and position_id='{}'
					""".
					format(
						account_number, select_currency, position_id
						))

		cursor.execute(query)
		cnx.commit()

		cursor.close()

		return True

def monitor_positions(account_number, select_currency, delta_map, avg_prices, pair_bid_ask_map, delta_window, is_variable_stop_loss):

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	
	print (account_type)
	response_value, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/summary", "GET")
	j = json.loads(response_value)

	print (j)
	account_profit = float(j['account'][u'unrealizedPL'])
	account_balance = float(j['account'][u'balance'])
	margin_available = float(j['account']['marginAvailable'])
	margin_used = float(j['account']['marginUsed'])

	total_balance += account_balance
	total_float_profit += account_profit
	total_margin_available += margin_available
	total_margin_used += margin_used


	cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

	cursor = cnx.cursor()

	query = ("""
		SELECT pair_order_id, pair, is_revert, revert_count, order_amount, position_id, is_buy
		from stat_arb_single
		where is_close=0 and account_nbr='{}' 
		""".format(account_number))

	print (query)
	cursor.execute(query)

	orders, total_margin = get_open_trades(account_number, 0)
	open_order_ids = [order.order_id for order in orders]
	total_positions = len(orders)

	order_ids_map = {}
	order_ids_position_id_map = {}
	for row1 in cursor:
		pair_order_id = row1[0]
		pair = row1[1]
		is_revert = row1[2]
		revert_count = row1[3]
		order_amount = row1[4]
		position_id = row1[5]
		is_buy = row1[6]

		order_ids_map[pair_order_id] = is_buy
		order_ids_position_id_map[pair_order_id] = position_id

	new_orders = []
	for order in orders:
		order.position_id = order_ids_position_id_map[order.order_id]

		if order.order_id not in order_ids_map:
			trade_logger.info("Close Order Not Exist {}".format(order.order_id))
			close_order(account_number, order.order_id)
		else:
			new_orders.append(order)

			if order.dir != order_ids_map[order.order_id]:
				trade_logger.info("Order Wrong Direction {}".format(order.order_id))
				print ("wrong")
				sys.exit(0)

	orders = new_orders

	query = ("""
		SELECT pair, open_timestamp, base_currency, position_id, pair_order_id, is_revert, revert_count, is_buy, open_delta, order_amount
		from stat_arb_single
		where is_close=0 and account_nbr='{}' and base_currency='{}'
		""".format(account_number, select_currency))

	print (query)
	cursor.execute(query)

	setup_rows = []
	for row1 in cursor:
		setup_rows.append(row1)

	order_ids = []
	total_positions = 0
	max_open_time_stamp = 0
	for row1 in setup_rows:
		pair = row1[0]
		timestamp = row1[1]
		select_currency_temp = row1[2]
		position_id = row1[3]
		pair_order_id = row1[4]
		is_revert = row1[5]
		revert_count = row1[6]
		is_buy = row1[7]

		if pair_order_id in open_order_ids:
			total_positions += revert_count

		order_ids.append(pair_order_id)
		max_open_time_stamp = max(max_open_time_stamp, timestamp)

	orders = [order for order in orders if order.order_id in order_ids]

	total_pnl = 0
	float_pip_profit = 0
	for order in orders:
		pip_size = 0.0001
		if "JPY" in order.pair:
			pip_size = 0.01

		if order.dir == (avg_prices[order.pair] > order.open_price):
			pip_profit = abs(avg_prices[order.pair] - order.open_price)
		else:
			pip_profit = -abs(avg_prices[order.pair] - order.open_price)

		pip_profit /= pip_size

		if is_variable_stop_loss and pip_profit < -max(5,50 * min(1, abs(delta_map[order.pair]))):
			close_order(account_number, order.order_id)
			continue

		pip_profit -= 3
		float_pip_profit += pip_profit
		total_pnl += order.PnL

	trade_logger.info("Float Pip Profit {}".format(float_pip_profit))

	if total_positions > 0 and float_pip_profit > 400 and total_pnl > 0:
		
		trade_logger.info("Close Positive Pips {}, Profit {}".format(float_pip_profit, total_pnl))
		total_positions = 0

		for row1 in setup_rows:
			pair = row1[0]
			timestamp = row1[1]
			select_currency_temp = row1[2]
			position_id = row1[3]
			pair_order_id = row1[4]

			print ("closing")
			close_order(account_number, pair_order_id)

		cursor = cnx.cursor()

		query = ("""update stat_arb_single set is_close=1 
			where 
			account_nbr='{}' and base_currency='{}'
			""".
			format(
				account_number, select_currency
				))

		cursor.execute(query)
		cnx.commit()
	else:

		count = 0
		for row1 in setup_rows:
			pair = row1[0]
			timestamp = int(row1[1])
			select_currency_temp = row1[2]
			position_id = row1[3]
			pair_order_id = row1[4]
			is_revert = row1[5]
			reverse_count = row1[6]
			is_buy = row1[7]
			open_delta = row1[8]
			order_amount = row1[9]

			if calculate_time_diff(time.time(), timestamp) < 24:
				continue

			if pair_bid_ask_map[pair]['spread'] > 5:
				continue

			query = ("""update stat_arb_single set open_timestamp='{}' where position_id='{}'""".
				format(
					time.time(),
					position_id
					))

			cursor.execute(query)
			cnx.commit() 


			if pair_order_id not in open_order_ids:
				trade_logger.info("Revert Trade, Position ID: " + str(position_id))

				open_reverse_trade(pair_order_id, pair, is_revert == False, revert_count * 2, order_amount * 1.5, account_number, select_currency, avg_prices, is_buy == False, open_delta, position_id)

		return total_balance, account_profit, j, total_positions, orders

	cursor.close()

	return total_balance, account_profit, j, total_positions, orders

def get_avg_prices():

	avg_prices = {}
	pair_bid_ask_map = {}
	for pair in currency_pairs:
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
		pair_bid_ask_map[pair]['spread'] = curr_spread

		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[first_currency + "_" + second_currency] = curr_price

	for pair in currency_pairs:
		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[second_currency + "_" + first_currency] = 1.0 / avg_prices[pair]
		pair_bid_ask_map[second_currency + "_" + first_currency] = {}
		pair_bid_ask_map[second_currency + "_" + first_currency]['spread'] = pair_bid_ask_map[pair]['spread']

	return avg_prices, pair_bid_ask_map

def open_position(account_numbers, max_equity, select_currency, before_all_price_df, \
	delta_map, max_pair, avg_prices, pair_bid_ask_map, threshold_delta, is_revert_entry, \
	total_margin_used, total_margin_available, open_orders):

	if select_currency not in ["ALL", "NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:
		print ("Arg 2 Must Be Currency")
		sys.exit(0)

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
									  host='mysql.newscaptial.com',
									  database='newscapital')

	for pair in currency_pairs:

		if select_currency != "ALL" and select_currency not in pair:
			continue

		if pair_bid_ask_map[pair]['spread'] > 5:
			continue

		if abs(delta_map[pair]) < threshold_delta:
			continue

		cursor = cnx.cursor()

		print ("open pair", pair)

		query = ("""SELECT max(open_timestamp) as last_updated, count(*) as total from stat_arb_single where base_currency = '{}' and pair = '{}' and is_close=0 and account_nbr='{}' and revert_count=1 """.
			format(
				select_currency,
				pair,
				account_numbers[0]
				))

		print (query)
		cursor.execute(query)

		is_found = False
		for row1 in cursor:
			if int(row1[1]) > 0 and calculate_time_diff(time.time(), int(row1[0])) < 24:
				print ("order not open long enough", pair, calculate_time_diff(time.time(), int(row1[0])))
				is_found = True
				break

		if is_found:
			continue

		is_overexposed = False
		if len(open_orders) > 0 and (float(total_margin_available) / (total_margin_used + total_margin_available)) < 0.15:
			max_order = sorted(open_orders, key=lambda x: x.PnL)[-1]
			trade_logger.info("Overexposed {}".format(select_currency))

			cursor = cnx.cursor()

			query = ("""update stat_arb_single set is_close=1 where position_id='{}'""".
				format(
					max_order.position_id
					))

			cursor.execute(query)
			cnx.commit() 

			close_order(account_numbers[0], max_order.order_id)
			is_overexposed = True

		if is_revert_entry:
			is_buy = delta_map[pair] > 0
		else:
			is_buy = delta_map[pair] < 0

		print ("opening order", pair, is_buy)
		order_id, base_amount = enter_group_trades(select_currency, pair, avg_prices[pair], avg_prices, account_numbers, is_buy, max_equity, None, delta_map[pair], 1)
		if order_id is not None:
			print ("insert order")

			cnx = mysql.connector.connect(user='andstv48', password='Password81',
									  host='mysql.newscaptial.com',
									  database='newscapital')

			cursor = cnx.cursor()

			query = ("""INSERT INTO stat_arb_single(user_id, pair, is_buy, open_timestamp, base_currency, pair_order_id, account_nbr, open_delta, is_revert, revert_count, order_amount)
				values ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')""".
				format(
					1,
					pair,
					1 if is_buy else 0,
					time.time(),
					select_currency,
					order_id,
					account_numbers[0],
					delta_map[pair], 
					0,
					1,
					base_amount
					))

			print (query)

			cursor.execute(query)
			cnx.commit()

			cursor.close()

		# Only one new trade per hour if overexposed
		if is_overexposed:
			break


	return False

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

def create_correlation_graph(before_all_price_df, back_days):


    times = before_all_price_df["times"].values.tolist()

    delta_window = []
    days_back_range = 20

    sorted_set = []
    for index in range(len(times) - (back_days * 24) - 1, len(times), 24):

    	print (index, len(times))

        sorted_pairs = []
        for j, currency_pair in enumerate(currency_pairs):

            delta_map = {}
            correlation_map = {}
            for select_currency in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:

                if select_currency not in currency_pair:
                    continue

                delta_map[select_currency] = {}
                correlation_map[select_currency] = {}

                for days_back in range(2, days_back_range):
                    hours_back = days_back * 24

                    z_scores = []
                    z_score_map = {}

                    pair2 = before_all_price_df["prices" + str(j)].values.tolist()[-hours_back+index:index]
  
                    if currency_pair[0:3] != select_currency:
                        pair2 = [1.0 / price for price in pair2]

                    correlations = []
                    for i, compare_pair in enumerate(currency_pairs):
                        if select_currency not in compare_pair:
                            continue

                        pair1 = before_all_price_df["prices" + str(i)].values.tolist()[-hours_back+index:index]
                        if compare_pair[0:3] != select_currency:
                            pair1 = [1.0 / price for price in pair1]

                        if i != j:
                            correlation, p_value = stats.pearsonr(pair1, pair2)
                            correlations.append(correlation)

                        mean = np.mean(pair1)
                        std = np.std(pair1)

                        z_score = (pair1[-1] - mean) / std 
                        z_scores.append(z_score)

                        z_score_map[compare_pair] = z_score

                    mean_correlation = np.mean(correlations)
                    mean_z_score = np.mean(z_scores)
                    delta = z_score_map[currency_pair] - mean_z_score

                    if currency_pair[0:3] != select_currency:
                        delta = -delta

                    delta_map[select_currency][days_back] = delta
                    correlation_map[select_currency][days_back] = mean_correlation


            matrix = np.zeros((days_back_range, days_back_range))

            diffs = []
            diff_left = []
            diff_right = []
            for i in range(2, days_back_range):
                for j in range(2, days_back_range):

                    delta_i = delta_map[currency_pair[0:3]][i]
                    delta_j = delta_map[currency_pair[4:7]][j]

                    correlation_i = max(0, correlation_map[currency_pair[0:3]][i])
                    correlation_j = max(0, correlation_map[currency_pair[4:7]][j])

                    if correlation_i + correlation_j > 0:
                        matrix[i][j] = ((delta_i * correlation_i) + (delta_j * correlation_j)) / (correlation_i + correlation_j)
                        diffs.append(matrix[i][j])
                        diff_left.append(delta_i)
                        diff_right.append(delta_j) 

            diffs = np.mean(diffs)

            sorted_pairs.append([currency_pair, diffs])

        sorted_set.append(sorted_pairs)

    return sorted_set



def monitor_accounts():
	
	count = 0
	for i, compare_pair in enumerate(currency_pairs):

		print (compare_pair)
		prices, times = get_time_series(compare_pair, 24 * 35)
		before_price_df2 = pd.DataFrame()
		before_price_df2["prices" + str(i)] = prices
		before_price_df2["times"] = times
		count += 1

		if count > 1:
			before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
			before_all_price_df.reset_index(inplace=True)
		else:
			before_all_price_df = before_price_df2


	cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

	cursor = cnx.cursor()
	query = ("""
		SELECT account_nbr, api_key, select_currency, max_equity, is_demo, last_updated, threshold_delta, is_revert_entry, is_variable_stop_loss
		from managed_accounts 
		where model_type = 'stat_basket'
		""")

	print (query)
	cursor.execute(query)

	sorted_sets = create_correlation_graph(before_all_price_df, 4)

	setup_rows = []
	for row1 in cursor:
		setup_rows.append(row1)

	pair_deltas = {}
	pair_ranks = {}
	global_ranks = {}
	for row1 in setup_rows:
		global api_key
		global trade_logger
		global account_type

		account_number = row1[0]
		api_key = row1[1]
		select_currency = row1[2]
		max_equity = row1[3]
		is_demo = row1[4]
		last_updated = row1[5]
		threshold_delta = row1[6]
		is_revert_entry = row1[7]
		is_variable_stop_loss = row1[8]

		global_deltas = sorted(sorted_sets[-1], key=lambda x: abs(x[1]), reverse=True)
		if select_currency != "ALL":
			deltas = [v for v in global_deltas if select_currency in v[0]]
		else:
			deltas = global_deltas

		delta = deltas[0][1]
		max_pair = deltas[0][0]

		delta_map = {}
		for i, item in enumerate(deltas):

			global_ranks[item[0]] = [j for j, v in enumerate(global_deltas) if v[0] == item[0]][0]
			delta_map[item[0]] = item[1]

			if item[0] not in pair_deltas:
				pair_deltas[item[0]] = []

			pair_deltas[item[0]].append(item[1])

			if item[0] not in pair_ranks:
				pair_ranks[item[0]] = []

			pair_ranks[item[0]].append(i)

		if select_currency != "ALL":
			delta_window = [max([abs(v[1]) for v in sorted_set if select_currency in v[0]]) for sorted_set in sorted_sets]
		else:
			delta_window = [max([abs(v[1]) for v in sorted_set]) for sorted_set in sorted_sets]
		

		print (select_currency, delta_window, max_pair, calculate_time_diff(time.time(), last_updated))

		trade_logger = setup_logger('first_logger', root_dir + "basket_trading_{}_{}.log".format(account_number, select_currency))

		trade_logger.info("Deltas " + str(delta_map))

		print (select_currency, calculate_time_diff(time.time(), last_updated))

		avg_prices, pair_bid_ask_map = get_avg_prices()
		
		if is_demo:
			account_type = "fxpractice"
		else:
			account_type = "fxtrade"

		equity, float_pnl, account, total_positions, open_orders = monitor_positions(account_number, select_currency, delta_map, avg_prices, pair_bid_ask_map, delta_window, is_variable_stop_loss)

		if total_positions > 0:
			margin_available = float(account['account']['marginAvailable'])
			margin_used = float(account['account']['marginUsed'])
		else:
			open_orders = []
			margin_used = 0
			margin_available = 0

		is_success = open_position([account_number], max_equity, select_currency, before_all_price_df, \
									 delta_map, max_pair, avg_prices, pair_bid_ask_map, threshold_delta, is_revert_entry, \
									 margin_used, margin_available, open_orders)


		account_profit = float(account['account'][u'unrealizedPL'])
		account_balance = float(account['account'][u'balance'])

		trade_logger.info("Equity: {}".format(account_balance + account_profit))
		trade_logger.info("PnL: {}".format(float_pnl))

		is_hedge_account = account['account'][u'hedgingEnabled']

		cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

		cursor = cnx.cursor()
		query = ("""
			UPDATE managed_accounts 
			set max_equity = '{}',
			float_pnl = '{}',
			account_value = '{}',
			is_hedged = '{}',
			account_currency = '{}',
			last_updated = '{}',
			margin_used = '{}',
			margin_available = '{}',
			total_positions = '{}',
			meta_json = '{}'
			where account_nbr = '{}' 
			and select_currency = '{}'
			""".format(max_equity, 
				float_pnl, 
				equity, 
				1 if is_hedge_account else 0, 
				account["account"]["currency"], 
				time.time(), 
				margin_used,
				margin_available,
				total_positions,
				json.dumps(delta_window),
				account_number,
				select_currency))

		print (query)
		cursor.execute(query)
		cnx.commit()


		handlers = trade_logger.handlers[:]
		for handler in handlers:
		    handler.close()
		    trade_logger.removeHandler(handler)



if get_mac() != 150538578859218:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 



monitor_accounts()

