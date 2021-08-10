import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
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
from lxml.html import fromstring

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

import mysql.connector
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
import psutil

import matplotlib
matplotlib.use('Agg')

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from bisect import bisect
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback

api_key = None
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
		self.calendar_time = 0
		self.barrier_size = 0
		self.carry_cost = 0
		self.ideal_price = 0
		

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

	print url, api_key


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

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""

	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger

if get_mac() != 150538578859218:
	root_dir = "/var/www/html" 
else:
	root_dir = "." 

trade_logger = setup_logger('first_logger', "{}/execute_portfolio.log".format(root_dir))

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

def get_open_trades(account_number, order_metadata, total_margin):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades?count=50"

	pair_spread = {}
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

			
			if pair not in pair_spread:
				response, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v1/prices?instruments=" + pair, "GET")
				response = json.loads(response)['prices']

				for price in response:
					if price['instrument'] == pair:
						curr_price = (price['bid'] + price['ask']) / 2
						curr_spread = abs(price['bid'] - price['ask']) / pip_size
						break

				pair_spread[pair] = curr_spread

			time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())
			time_diff_hours = calculate_time_diff(time.time(), time_start)

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair + "_" + str(amount > 0)
			order.dir = amount > 0
			order.order_id = int(order_id)
			order.account_number = account_number
			order.margin_used = margin_used
			order.PnL = PnL
			order.spread = pair_spread[pair]
			order.time_diff_weeks = time_diff_hours / (24 * 5)
			order.pair_only = pair

			orders.append(order)

	return orders, total_margin

def close_order(order):

	order_info, _ =  sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + order.account_number + "/trades/" + str(order.order_id) + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info))  
		return True

	return False

def create_order(open_price, pair, account_numbers, trade_dir, order_amount, exposure, orders_by_pair):

	if round(200 / exposure) < 6:
		return None

	account_number = account_numbers[0]

	order_sizes = []
	for pair_temp in orders_by_pair:
		order_sizes += [order.amount for order in orders_by_pair[pair_temp]]

	mean = np.mean(order_sizes)
	std = np.std(order_sizes)
	order_amount = int(round(min(mean + (std * 2), order_amount)))

	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01
 

	precision = '%.4f'
	if pair[4:7] == 'JPY':
		precision = '%.2f'

	if trade_dir == True:
		take_profit_price = open_price + (pip_size * max(10, round(200 / exposure)))
		stop_loss_price = open_price - (pip_size * max(10, round(100 / exposure)))
		tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % take_profit_price) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
		order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ',' + tp_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		take_profit_price = open_price - (pip_size * max(10, round(200 / exposure)))
		stop_loss_price = open_price + (pip_size * max(10, round(100 / exposure)))
		tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % take_profit_price) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
		order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ',' + tp_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		

	return order_info

def draw_pnl_plot(orders, orders_by_pair, orders_by_currency, exposure):

	with PdfPages('{}/portfolio.pdf'.format(root_dir)) as pdf:
		import numpy as np
		orders = sorted(orders, key=lambda x: -x.PnL, reverse=True)

		pnl_mean = np.mean([order.PnL for order in orders])
		pnl_std = np.std([order.PnL for order in orders])

		z_scores = []
		pairs = []
		for order in orders:
			z_scores.append((order.PnL - pnl_mean) / pnl_std)
			pairs.append(order.pair.replace("_True", " (BUY)").replace("_False", " (SELL)"))
			print (order.pair, z_scores[-1])


		# Example data
		people = pairs
		y_pos = np.arange(len(people))
		performance = z_scores

		plt.figure(figsize=(11,10))
		plt.axvline(x=3 / exposure, color='green')
		plt.axvline(x=-2 / exposure, color='red')
		colors = ['green' if i >= len(pairs) * 0.5 else 'red' for i in range(len(pairs))]

		plt.barh(y_pos, performance, align='center', alpha=0.4, color=colors)
		plt.yticks(y_pos, people)
		plt.xlabel('PnL Z-Score')
		plt.title('Position PnL Z-Score')
		pdf.savefig()
		plt.close()

		mean = np.mean([abs(orders_by_pair[pair]) for pair in orders_by_pair])
		std = np.std([abs(orders_by_pair[pair]) for pair in orders_by_pair])

		z_scores = []
		pairs = []
		for pair in orders_by_pair:
			z_scores.append([pair.replace("_True", " (BUY)").replace("_False", " (SELL)"), (abs(orders_by_pair[pair]) - mean) / std])

		z_scores = sorted(z_scores, key=lambda x: x[1])

		# Example data
		people = [v[0] for v in z_scores]
		y_pos = np.arange(len(people))
		performance = [v[1] for v in z_scores]
		colors = ['green' if z_score < 1.5 else 'red' for z_score in performance]

		plt.figure(figsize=(11,10))
		plt.axvline(x=1.5, color='red')
		plt.barh(y_pos, performance, align='center', alpha=0.4, color=colors)
		plt.yticks(y_pos, people)
		plt.xlabel('Amount Z-Score')
		plt.title('Position Amount Z-Score')
		pdf.savefig()
		plt.close()


		mean = np.mean([abs(orders_by_currency[currency]) for currency in orders_by_currency])
		std = np.std([abs(orders_by_currency[currency]) for currency in orders_by_currency])

		z_scores = []
		pairs = []
		for currency in orders_by_currency:
			trade_dir = "BUY" if orders_by_currency[currency] > 0 else "SELL"
			z_scores.append([currency + " ({})".format(trade_dir), (abs(orders_by_currency[currency]) - mean) / std])

		z_scores = sorted(z_scores, key=lambda x: x[1])

		# Example data
		people = [v[0] for v in z_scores]
		y_pos = np.arange(len(people))
		performance = [v[1] for v in z_scores]
		colors = ['green' if z_score < 2 else 'red' for z_score in performance]

		plt.figure(figsize=(11,10))
		plt.axvline(x=2, color='red')
		plt.barh(y_pos, performance, align='center', alpha=0.4, color=colors)
		plt.yticks(y_pos, people)
		plt.xlabel('Amount Z-Score')
		plt.title('Currency Amount Z-Score')
		pdf.savefig()
		plt.close()



def process_pending_trades(account_numbers):

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

	trade_logger.info('Equity: ' + str(total_balance + total_float_profit))

	orders = []
	for account_number in account_numbers:
		orders1, total_margin = get_open_trades(account_number, {}, total_margin_used)
		orders += orders1

	orders_by_pair = {}
	actual_order_by_pair = {}
	orders_by_currency = {}
	pnl_by_pair = {}
	for order in orders:
		if order.pair not in orders_by_pair:
			orders_by_pair[order.pair] = []
			pnl_by_pair[order.pair] = 0
			actual_order_by_pair[order.pair] = []

		if order.dir:
			orders_by_pair[order.pair].append(order.amount)
		else:
			orders_by_pair[order.pair].append(-order.amount)

		currency1 = order.pair[0:3]
		currency2 = order.pair[4:7]

		pnl_by_pair[order.pair] += order.PnL
		actual_order_by_pair[order.pair].append(order)


		if currency1 not in orders_by_currency:
			orders_by_currency[currency1] = []

		if currency2 not in orders_by_currency:
			orders_by_currency[currency2] = []

		if order.dir:
			orders_by_currency[currency1].append(order.amount)
			orders_by_currency[currency2].append(-order.amount)
		else:
			orders_by_currency[currency1].append(-order.amount)
			orders_by_currency[currency2].append(order.amount)

	if len(orders) == 0:
		return

	print ("total orders", len(orders))


	pnl_mean = np.mean([abs(pnl_by_pair[pair]) for pair in pnl_by_pair])
	pnl_std = np.std([abs(pnl_by_pair[pair]) for pair in pnl_by_pair])

	pnl_abs_mean = np.mean([abs(pnl_by_pair[pair]) for pair in pnl_by_pair])

	exposure = ((margin_used / total_balance) / 0.02)
	trade_logger.info('Exposure: ' + str(exposure))

	for pair in pnl_by_pair:
		z_score = (pnl_by_pair[pair] - pnl_mean) / pnl_std

		if z_score > (3 / exposure):

			min_order = actual_order_by_pair[pair][0]
			if min_order.spread < 5:
				for order in actual_order_by_pair[pair]:
					close_order(order)

					if order.amount < min_order.amount:
						min_order = order

				trade_logger.info('Close ALL : ' + str(pair))
				order_info = create_order(min_order.open_price, min_order.pair_only, account_numbers, min_order.dir, int(min_order.amount), 1, actual_order_by_pair)
				trade_logger.info('FINISH Close ALL : ' + str(pair))

	pnl_mean = np.mean([abs(order.PnL) for order in orders])
	pnl_std = np.std([abs(order.PnL) for order in orders])

	pnl_abs_mean = np.mean([abs(order.PnL) for order in orders])

	for order in orders:
		z_score = (order.PnL - pnl_mean) / pnl_std
		
		if abs(order.PnL) / abs(pnl_abs_mean) > (4 / exposure):

			if z_score > (3 / exposure):
				trade_logger.info("Close Order, " + str(order.pair) + ", z_score, " + str(z_score) + ", max_z_score, " + str(3 / exposure), ", current_wait, " + str(order.time_diff_weeks) + "max wait, " + str((1 / exposure) / abs(z_score)))

				close_order(order)

	for currency in orders_by_currency:
		orders_by_currency[currency] = sum(orders_by_currency[currency])

	for pair in orders_by_pair:
		orders_by_pair[pair] = sum(orders_by_pair[pair])

	draw_pnl_plot(orders, orders_by_pair, orders_by_currency, exposure)

	# First we select the best order from each pair - optimistic
	orders = sorted(orders, key=lambda x: x.PnL, reverse=True)

	new_orders = []
	order_pairs = set()
	for order in orders:
		if order.pair in order_pairs:
			continue

		new_orders.append(order)
		order_pairs.add(order.pair)

	orders = new_orders

	# Then we rank the best for each pair in ascending order to find percentile
	orders = sorted(orders, key=lambda x: x.PnL, reverse=False)

	new_orders = []
	whilte_list_orders = []
	for i, order in enumerate(orders):

		mean = np.mean([abs(orders_by_pair[pair]) for pair in orders_by_pair])
		std = np.std([abs(orders_by_pair[pair]) for pair in orders_by_pair])

		print ("pair amount", order.pair, (abs(orders_by_pair[order.pair]) - mean) / std,  orders_by_pair[order.pair], mean, std)

		if (abs(orders_by_pair[order.pair]) - mean) / std > 1.5 and (order.dir == (orders_by_pair[order.pair] > 0)):
			continue

		found = True
		currency1 = order.pair[:3]
		currency2 = order.pair[4:7]

		for select_currency in [currency1, currency2]:
			mean = np.mean([abs(orders_by_currency[currency]) for currency in orders_by_currency])
			std = np.std([abs(orders_by_currency[currency]) for currency in orders_by_currency])

			print ("currency amount", select_currency, order.pair, (abs(orders_by_currency[select_currency]) - mean) / std )

			if order.pair[0:3] == select_currency:
				if (abs(orders_by_currency[select_currency]) - mean) / std > 2 and (orders_by_currency[select_currency] > 0) == order.dir:
					print ("fail1")
					found = False
			else:
				if (abs(orders_by_currency[select_currency]) - mean) / std > 2 and (orders_by_currency[select_currency] > 0) != order.dir:
					print ("fail2")
					found = False

		if found:
			whilte_list_orders.append(order)

	for i, order in enumerate(whilte_list_orders):
		order_frac = float(i + 1) / len(whilte_list_orders)

		if order.spread < 5:

			order_amount = max(1, round(order.amount * (2 * order_frac)))

			if order_frac > 0.5:
				order_amount = max(order.amount + 1, order_amount)
			else:
				order_amount = order.amount

			
			exposure = max(1, (abs(orders_by_pair[order.pair]) / total_balance) / 0.05)

			print ("exposure", exposure, "*********", abs(orders_by_pair[order.pair]) )

			print ("execute", order.pair, order_frac)
			order_info = create_order(order.open_price, order.pair_only, account_numbers, order.dir, int(order_amount), exposure, actual_order_by_pair)
			print (order_info)
			

if is_valid_trading_period(time.time()) == False:
	sys.exit(0)

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

checkIfProcessRunning('execute_update_portfolio.py', '')



cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

cursor = cnx.cursor()
query = ("SELECT account_nbr, api_key FROM managed_accounts t2 where t2.user_id=61 and t2.account_nbr='001-011-2949857-007'")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
	setup_rows.append(row1)

cursor.close()

for row in setup_rows:

	account_nbr = row[0]
	api_key = row[1]

	process_pending_trades([account_nbr]) #demo


