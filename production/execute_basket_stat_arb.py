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

api_key = "8bde4f67a710b42553a821bdfff8efa9-eb1cb834f4060df9949504beb3356265"
file_ext_key = ""
account_type = "fxtrade"

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
	next_link = "https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades?count=50"

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
				order_info, _ = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
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
			order.open_time = trade['openTime']
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders, total_margin

def close_order(account_number, order_id):

	if order_id == -1:
		return True


	order_info, _ =  sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades/" + str(order_id) + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		return True

	return False

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
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (140 * pip_size))) + '"}'
		#gsl_price = '"guaranteedStopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (170 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (140 * pip_size))) + '"}'
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


def close_group_trades(orders):


	total_pnl = 0

	for order in orders:
		total_pnl += order.PnL


	for order in orders:

		if total_pnl > 0: 
			order_metadata, is_success = close_order(order)
			if is_success == True:
				continue



def enter_group_trades(pair, curr_price, avg_prices, account_numbers, is_buy, account_balance):


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
	new_order.base_amount = (account_balance / 5000) * 5000 

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
		order_id = int(order_info['orderFillTransaction']['id'])

		new_order.order_id = order_id
		new_order.amount = order_amount
		new_order.account_number = account_number

		trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 

	else:
		order_id = None
		trade_logger.info('Order Error: ' + str(order_info)) 

	return order_id

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


def close_position(account_number, position_id):

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

	cursor = cnx.cursor()
	query = ("""SELECT pair1_order_id, pair2_order_id from stat_arb_pairs where position_id='{}'""".
		format(
			position_id
			))

	print (query)
	cursor.execute(query)

	for row1 in cursor:
		order_info, _ =  sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades/" + str(row1[0]) + "/close", "PUT")
		order_info, _ =  sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades/" + str(row1[1]) + "/close", "PUT")

	cursor = cnx.cursor()

	query = ("""update stat_arb_pairs set is_close=1 where position_id='{}'""".
		format(
			position_id
			))

	cursor.execute(query)
	cnx.commit()

	cursor.close()

def monitor_positions(account_number):

	count = 0
	for i, compare_pair in enumerate(currency_pairs):

		print (compare_pair)
		prices, times = get_time_series(compare_pair, 24 * 30 * 3)
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
	query = ("""SELECT pair1, pair2, open_timestamp, base_currency, position_id, pair1_order_id, pair2_order_id 
		from stat_arb_pairs 
		where is_close=0 and account_nbr=''
		""")

	print (query)
	cursor.execute(query)

	setup_rows = []
	for row1 in cursor:
		setup_rows.append(row1)

	for row1 in setup_rows:
		pair1 = row1[0]
		pair2 = row1[1]
		timestamp = row1[2]
		select_currency = row1[3]
		position_id = row1[4]
		pair1_order_id = row1[5]
		pair2_order_id = row1[6]

		subset_df = before_all_price_df[before_all_price_df["times"] >= timestamp]

		basket_prices = []
	
		for i, compare_pair in enumerate(currency_pairs):
			if select_currency in compare_pair:

				prices = subset_df["prices" + str(i)].values.tolist()

				select_pair = compare_pair
				if compare_pair[0:3] != select_currency:
					prices = [1.0 / price for price in prices]
					select_pair = compare_pair[4:7] + "_" + compare_pair[0:3]

				if select_pair == pair1:
					pair1_index = len(basket_prices)

				if select_pair == pair2:
					pair2_index = len(basket_prices)

				basket_prices.append(prices)

		print (pair1_index, pair2_index)

		correlations = []
		for i in range(len(basket_prices)):
			for j in range(i + 1, len(basket_prices)):

				correlation, p_value = stats.pearsonr(basket_prices[i], basket_prices[j])

				correlations.append(correlation)

		mean_price1 = basket_prices[pair1_index][-1] - np.mean(basket_prices[pair1_index])
		pip_size = 0.0001
		if "JPY" in pair1 and select_currency != "JPY":
			pip_size = 0.01

		mean_price1 /= pip_size

  
		mean_price2 = basket_prices[pair2_index][-1] - np.mean(basket_prices[pair2_index])
		pip_size = 0.0001
		if "JPY" in pair2 and select_currency != "JPY":
			pip_size = 0.01

		mean_price2 /= pip_size
		total_margin = 0
		orders, total_margin = get_open_trades(account_number, total_margin)
		float_pnl = sum([order.PnL for order in orders if order.order_id in [pair1_order_id, pair2_order_id]])

		cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

		print ("update")
		cursor = cnx.cursor()

		query = ("""update stat_arb_pairs set pnl='{}', correlation='{}', pair1_price_diff = '{}', pair2_price_diff = '{}' where position_id='{}'""".
			format(
				float_pnl,
				np.mean(correlations),
				mean_price1, 
				mean_price2,
				position_id
				))

		cursor.execute(query)
		cnx.commit()

		if mean_price1 > mean_price2 or np.mean(correlations) < 0.35 or (time.time() - timestamp) > 60 * 60 * 24 * 30:
			print ("closing")
			close_position(account_number, pair1_order_id)
			close_position(account_number, pair2_order_id)

			cursor = cnx.cursor()

			query = ("""update stat_arb_pairs set is_close=1 where position_id='{}'""".
				format(
					position_id
					))

			cursor.execute(query)
			cnx.commit()

		cursor.close()


	cursor.close()

def open_position(account_numbers):

	select_currency = sys.argv[2]
	if select_currency not in ["NZD", "AUD", "USD", "CAD", "JPY", "GBP", "EUR", "CHF"]:
		print ("Arg 2 Must Be Currency")
		sys.exit(0)

	trade_dir = sys.argv[4]
	if trade_dir == "BUY":
		trade_dir = True
	elif trade_dir == "SELL":
		trade_dir = False
	else:
		print ("Trade Dir Must Be BUY or SELL")
		sys.exit(0)

	if len(sys.argv) > 5:
		exclude_pairs = sys.argv[5].split(",")
		for pair in exclude_pairs:
			if pair not in currency_pairs:
				print ("Is Not A Currency Pair", pair)
				sys.exit(0)
	else:
		exclude_pairs = []

	print (account_numbers[0], api_key)
	response_value, _ = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_numbers[0] + "/summary", "GET")
	j = json.loads(response_value)


	account_profit = float(j['account'][u'unrealizedPL'])
	account_balance = float(j['account'][u'balance'])
	margin_available = float(j['account']['marginAvailable'])
	margin_used = float(j['account']['marginUsed'])

	timestamp = calendar.timegm(datetime.datetime.strptime(sys.argv[3], "%Y-%m-%d").timetuple())
	hours = int((time.time() - timestamp) / (60 * 60))


	count = 0
	for i, compare_pair in enumerate(currency_pairs):
		if select_currency not in compare_pair:
			continue

		if compare_pair in exclude_pairs:
			continue

		print (compare_pair)
		prices, times = get_time_series(compare_pair, hours+ 24)
		before_price_df2 = pd.DataFrame()
		before_price_df2["prices" + str(i)] = prices
		before_price_df2["times"] = times
		count += 1

		if count > 1:
			before_all_price_df = before_all_price_df.set_index('times').join(before_price_df2.set_index('times'), how='inner')
			before_all_price_df.reset_index(inplace=True)
		else:
			before_all_price_df = before_price_df2

	before_all_price_df = before_all_price_df[before_all_price_df["times"] >= timestamp]

	print ("timestamp:", timestamp)
	print (before_all_price_df.head(5))

	deltas = []
	basket_prices = []
	for i, compare_pair in enumerate(currency_pairs):
		if select_currency not in compare_pair:
			continue

		if compare_pair in exclude_pairs:
			continue

		pip_size = 0.0001
		if "JPY" in compare_pair[4:7] and select_currency != "JPY":
			pip_size = 0.01

		prices = before_all_price_df["prices" + str(i)].values.tolist()

		select_pair = compare_pair
		if compare_pair[0:3] != select_currency:
			prices = [1.0 / price for price in prices]
			select_pair = compare_pair[4:7] + "_" + compare_pair[0:3]

		mean_price = np.mean(prices)
		basket_prices.append(prices)
		deltas.append([(prices[-1] - mean_price) / pip_size, select_pair])

	correlations = []
	for i in range(len(basket_prices)):
		for j in range(i + 1, len(basket_prices)):

			correlation, p_value = stats.pearsonr(basket_prices[i], basket_prices[j])
			correlations.append(correlation)

	if np.mean(correlations) < 0.5:
		print ("Low Correlation", np.mean(correlations))
		return

	deltas = sorted(deltas, key=lambda x: x[0])
	pair1 = deltas[0][1]
	pair2 = deltas[-1][1]

	if trade_dir:
		select_pair = pair1
	else:
		select_pair = pair2

	print (pair1, pair2)

	avg_prices = {}
	pair_bid_ask_map = {}
	for pair in currency_pairs:
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

	order_ids = []
	for pair in [pair1, pair2]:

		if select_pair != pair:
			order_ids.append(-1)
			continue

		is_buy = (pair in currency_pairs) == (pair == pair1)

		if pair in currency_pairs:
			actual_pair = pair
		else:
			actual_pair = pair[4:7] + "_" + pair[0:3]

		print ("opening order", pair, is_buy)
		order_id = enter_group_trades(actual_pair, avg_prices[actual_pair], avg_prices, account_numbers, is_buy, account_balance)
		order_ids.append(order_id)

	if order_ids[0] is not None and order_ids[1] is not None:

		cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

		cursor = cnx.cursor()

		query = ("""SELECT * 
			from stat_arb_pairs 
			where is_close=0 and account_nbr='' and pair1='{}' and pair2='{}' and base_currency='{}'
			""".format(
				pair1,
				pair2,
				select_currency
				))

		print (query)
		cursor.execute(query)

		setup_rows = []
		for row1 in cursor:
			setup_rows.append(row1)

		if len(setup_rows) == 0:

			query = ("""INSERT INTO stat_arb_pairs(user_id, pair1, pair2, open_timestamp, base_currency, pair1_order_id, pair2_order_id, pair1_price_diff, pair2_price_diff, trade_timestamp, correlation)
				values ('{}','{}','{}','{}','{}','{}','{}','{}','{}','{}','{}')""".
				format(
					1,
					pair1,
					pair2,
					timestamp,
					select_currency,
					order_ids[0],
					order_ids[1],
					deltas[0][0],
					deltas[-1][0],
					time.time(),
					np.mean(correlations)
					))

			print (query)

			cursor.execute(query)
			cnx.commit()

			cursor.close()
		else:
			print ("same position already open")





accounts = [
	["101-011-9454699-002", "101-011-9454699-003"],
	["001-011-2949857-007", "001-011-2949857-007"],
	["101-011-14392464-002", "101-011-14392464-002"],
]


if get_mac() != 150538578859218:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 

trade_logger = setup_logger('first_logger', root_dir + "stat_arb_trading.log")



try:

	if sys.argv[1] == "open":
		open_position(accounts[1])
	elif sys.argv[1] == "close":
		close_position(accounts[1][0], sys.argv[2])
	elif sys.argv[1] == "monitor":
		monitor_positions(accounts[1][0])

	trade_logger.info('Finished ') 
except:
	print (traceback.format_exc())
	trade_logger.info(traceback.format_exc())


