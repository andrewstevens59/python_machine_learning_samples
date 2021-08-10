import sys
import math
from datetime import datetime
from random import *
import os.path


import pickle

import pycurl
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
		self.pnl = 0
		self.tp_price = 0
		self.sl_price = 0
		self.actual_amount = 0
		self.account_number = None
		self.time_diff_hours = 0
		self.order_id = 0


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


def get_open_trades(account_number):

	trades = []
	response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades?count=500", "GET")

	j = json.loads(response_value)
	open_orders = j['trades']

	orders = []
	pair_time_diff = {}
	for trade in open_orders:

		print trade

		#print trade['openTime']
		s = trade['openTime'].replace(':', "-")
		s = s[0 : s.index('.')]
		order_id = trade['id']
		open_price = float(trade[u'price'])
		pair = trade[u'instrument']
		amount = float(trade[u'currentUnits'])
		pair = trade['instrument'].replace("/", "_")
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

		if week_day_start != 6 and week_day_end < week_day_start:
			time_diff_hours -= 49
		elif week_day_end == 6 and week_day_end > week_day_start:
			time_diff_hours -= 49

		order = Order()
		order.open_price = open_price
		order.amount = abs(amount)
		order.pair = pair
		order.dir = amount > 0
		order.time_diff_hours = time_diff_hours
		order.order_id = order_id
		order.account_number = account_number
		orders.append(order)

	return orders

def process_pending_trades(account_numbers):


	balance = 0
	float_profit = 0
	for account_number in account_numbers:
		response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/", "GET")
		j = json.loads(response_value)
		account_profit = float(j['account'][u'unrealizedPL'])
		account_balance = float(j['account'][u'balance'])
		margin_used = float(j['account'][u'marginUsed'])

		balance += account_balance
		float_profit += account_profit

	logging.info('Equity: ' + str(balance + float_profit))

	orders = []
	for account_number in account_numbers:
		orders += get_open_trades(account_number)

	orders_by_pair = {}
	time_diff_hours = {}
	for pair in currency_pairs:
		orders_by_pair[pair] = []
		time_diff_hours[pair] = 9999999999

	for order in orders:
		orders_by_pair[order.pair].append(order)
		time_diff_hours[order.pair] = min(time_diff_hours[order.pair], order.time_diff_hours)

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))
	model_pair_wt = pickle.load(open("/tmp/model_pair_wt", 'rb'))

	if os.path.isfile("/tmp/group_metadata"):
		group_metadata = pickle.load(open("/tmp/group_metadata", 'rb'))
	else:
		group_metadata = {}

	model_predictions = {}
	for pair in currency_pairs:

		response = sendCurlRequest("https://api-fxtrade.oanda.com/v1/prices?instruments=" + pair, "GET")
		response = json.loads(response)['prices']

		spread = avg_spreads[pair]

		first_currency = pair[0:3]
		second_currency = pair[4:7]

		if second_currency != "USD":
			pair_mult = avg_prices[second_currency + "_USD"]

			if second_currency == "JPY":
				pair_mult *= 100
		else:
			pair_mult = 1.0

		pip_size = 0.0001
		if pair[4:] == "JPY":
			pip_size = 0.01

		for spread_count in range(1):
			curr_spread = 0
			for price in response:
				if price['instrument'] == pair:
					curr_price = (price['bid'] + price['ask']) / 2
					curr_spread = abs(price['bid'] - price['ask']) / pip_size
					break

			if curr_price == 0:
				print "price not found"
				continue
			if curr_spread < spread * 1.2:
				break

		last_processed_key = pair + "_last_processed"
		last_price_key = pair + "_last_price"
		if last_price_key not in group_metadata:
			group_metadata[last_price_key] = curr_price

		price_diff = abs(curr_price - group_metadata[last_price_key]) / pip_size
		price_diff /= 100
		curr_spread /= max(1, price_diff)
		
		if curr_spread > spread * 1.2:
			continue

		group_metadata[last_processed_key] = time.time()
		group_metadata[last_price_key] = curr_price
		
		if os.path.isfile("/tmp/model_predictions_" + pair) == False:
			continue

		model_predictions = pickle.load(open("/tmp/model_predictions_" + pair , 'rb'))

		for model in model_predictions:
			base_model_key = model['model_key'] 
			prediction = model['prediction']
			mean_price_change = model['mean_rate_change']
			confidence = max(model['confidence'])

			if base_model_key not in group_metadata:
				group_metadata[base_model_key + "_prev_dir"] = None

			print pair, prediction, mean_price_change

			for order in orders_by_pair[pair]:
				pip_diff = abs(curr_price - order.open_price) / pip_size
				if order.time_diff_hours >= 48:
					print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")
		

			if (prediction > 0) != prev_dir:
				group_metadata[base_model_key + "_reduce_amount"] = 1
			
			if abs(prediction) >= 2 and time_diff_hours[pair] >= 6 and group_metadata[base_model_key + "_reduce_amount"] < 8:

				prev_dir = group_metadata[base_model_key + "_prev_dir"]

				order_amount = int(round(((confidence * balance * 130 * model_pair_wt[base_model_key]) / (mean_price_change * pair_mult * group_metadata[base_model_key + "_reduce_amount"]))))
				print pair, order_amount, "(((((((((((", group_metadata[base_model_key + "_reduce_amount"], base_model_key
				if order_amount == 0:
					continue

				group_metadata[base_model_key + "_reduce_amount"] *= 2

				precision = '%.4f'
				if pair[4:] == 'JPY':
					precision = '%.3f'

				if prediction > 0:
					group_metadata[base_model_key + "_prev_dir"] = True
					tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (300 * pip_size))) + '"}'
					sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price - (300 * pip_size))) + '"}'

					if random.random() > 0.5:
						account_number = account_numbers[0]
					else:
						account_number = account_numbers[1]

					order_info = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + tp_price + ', ' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
				else:
					group_metadata[base_model_key + "_prev_dir"] = False
					tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price - (300 * pip_size))) + '"}'
					sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (300 * pip_size))) + '"}'

					if random.random() > 0.5:
						account_number = account_numbers[2]
					else:
						account_number = account_numbers[3]

					order_info = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + tp_price + ', ' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
				
				pickle.dump(group_metadata, open("/tmp/group_metadata", 'wb'))
				logging.info('New Order: ' + str(order_info)) 

	logging.info('Total Orders: ' + str(len(orders))) 
	pickle.dump(group_metadata, open("/tmp/group_metadata", 'wb'))


'''
if get_mac() == 154505288144005:

	print "Can't run locally - run only on remote server"

	
	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.get("/tmp/group_metadata", "/tmp/group_metadata")

	for pair in currency_pairs:
		sftp.get("/tmp/model_predictions_" + pair, "/tmp/model_predictions_" + pair)
'''

logging.basicConfig(filename='/tmp/trade.log',level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

#practice process_pending_trades(["101-001-9056476-001", "101-001-9056476-002"])
process_pending_trades(["001-001-1370090-004", "001-001-1370090-003"])


if get_mac() == 154505288144005:

	print "Can't run locally - run only on remote server"

	
	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.put("/tmp/model_pair_wt", "/tmp/model_pair_wt")
	sftp.put("/tmp/group_metadata", "/tmp/group_metadata")

	for pair in currency_pairs:
		sftp.put("/tmp/model_predictions_" + pair, "/tmp/model_predictions_" + pair)


