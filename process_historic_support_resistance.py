import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from pytz import timezone
import xgboost as xgb
from datetime import timedelta
import mysql.connector
from lxml.html import fromstring
from itertools import cycle
from scipy import stats
import datetime as dt
import traceback


import re

import matplotlib
matplotlib.use('Agg')

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import json
import copy

import math
import sys
import re

import numpy as np
import pandas as pd 
import pycurl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json


import os
from bisect import bisect

import paramiko
import json

import logging
import os
import enum

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback

currency_pairs = [
	"AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
	"AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
	"AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
	"AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
	"AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
	"CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
	"CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

def checkIfProcessRunning(processName, command):
	count = 0
	#Iterate over the all the running process
	for proc in psutil.process_iter():

		try:
			cmdline = proc.cmdline()

			# Check if process name contains the given name string.
			if len(cmdline) > 1 and processName.lower() in cmdline[1]: 
				count += 1
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			pass

	if count >= 2:
		sys.exit(0)

checkIfProcessRunning('draw_macro_trend_lines.py', "")

if get_mac() != 150538578859218:
	root_dir = "/root/trading/production/" 
else:
	root_dir = "" 


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


def load_time_series(symbol, year, is_bid_file):

	if get_mac() == 150538578859218:
		prefix = '/Users/andrewstevens/Downloads/economic_calendar/'
	else:
		prefix = '/root/trading_data/'

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir(prefix) if isfile(join(prefix, f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Candlestick_1_Hour_BID' in file:
			break

	if pair not in file:
		return None

	with open(prefix + file) as f:
		content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	from_zone = tz.gettz('America/New_York')
	to_zone = tz.tzutc()

	prices = []
	times = []
	volumes = []

	content = content[1:]

	if year != None:
		start_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".1.1 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())
		end_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".12.31 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())

	for index in range(len(content)):

		toks = content[index].split(',')
		utc = datetime.datetime.strptime(toks[0], "%d.%m.%Y %H:%M:%S.%f")

		time = calendar.timegm(utc.timetuple())

		if year == None or (time >= start_time and time < end_time):

			high = float(toks[2])
			low = float(toks[3])
			o_price = float(toks[1])
			c_price = float(toks[4])
			volume = float(toks[5])

			if high != low or volume > 0:
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes


def check_memory():
	import psutil
	import gc

	memory = psutil.virtual_memory() 
	while memory.percent > 80:
		gc.collect()
		memory = psutil.virtual_memory() 

# solve for a and b
def best_fit(X, Y):

	b, a = np.polyfit(X, Y, 1)

	return a, b

def store_model_prediction(pair, levels, curr_price):

	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	cursor = cnx.cursor()


	query = ("""INSERT INTO signal_summary(timestamp, pair, model_group, forecast_percentiles) 
				values (now(),'{}','{}','{}')""".
		format(
			pair,
			"Support And Resistance",
			json.dumps({"levels" : levels, "curr_price" : curr_price})
			))

	print (query)

	cursor.execute(query)
	cnx.commit()

def get_future_movement(a, b, time_offset, future_prices):


	future_time_periods = [1, 2, 3, 4, 5, 10, 15, 20]
	future_time_periods = [24 * t for t in future_time_periods]

	for period in future_time_periods:
		yfit = a + b * (time_offset + period)

		if period not in future_prices:
			future_prices[period] = []

		future_prices[period].append(yfit)

class Order():

	def __init__(self):
		self.amount = 0

def find_optimal_plan(price_levels, curr_price_index, orders, level, pnls):

	curr_price = price_levels[curr_price_index]

	for order in orders:

		if order.dir == (curr_price > order.open_price):
			order.pnl = abs(curr_price - order.open_price) * order.amount
		else:
			order.pnl = -abs(curr_price - order.open_price) * order.amount

	if level > 3:
		pnls += [order.pnl for order in orders]
		return pnls, None

	best_sharpe = None
	best_pnls = None

	best_decision_summary = None
	for order_dir in [True, False]:


		for order_amount in range(0, 4):

			new_orders = orders
			if order_amount > 0:
				order = Order()
				order.amount = order_amount
				order.dir = order_dir
				order.open_price = price_levels[curr_price_index]
				new_orders.append(order)

			right_prob = 1.0 / float(abs(curr_price - price_levels[curr_price_index + 1]))
			left_prob = 1.0 / float(abs(curr_price - price_levels[curr_price_index - 1]))
			total_sum = right_prob + left_prob

			right_prob /= total_sum
			left_prob /= total_sum

			right_pnls, right_decision = find_optimal_plan(price_levels, curr_price_index + 1, copy.deepcopy(new_orders), level + 1, [])
			left_pnls, left_decision = find_optimal_plan(price_levels, curr_price_index - 1, copy.deepcopy(new_orders), level + 1, [])
			new_pnls = [p * right_prob for p in right_pnls] + [p * left_prob for p in left_pnls] 

			if len(new_pnls) == 0:
				decision_sharpe = -1
			else:
				decision_sharpe = np.mean(new_pnls) / np.std(new_pnls)

			decision_summary = {
				"dir" : order_dir,
				"amount" : order_amount,
				"left_price" : price_levels[curr_price_index - 1],
				"right_price" : price_levels[curr_price_index + 1],
				"left_decision" : left_decision,
				"right_decision" : right_decision
			}

			if best_sharpe == None or decision_sharpe > best_sharpe:
				best_sharpe = decision_sharpe
				best_pnls = new_pnls
				best_decision_summary = decision_summary

	return best_pnls, best_decision_summary

def forecast_forward(best_decision, future_prices, weight):

	orders = []
	order = Order()
	order.dir = best_decision["dir"]
	order.amount = best_decision["amount"] * weight
	order.open_price = future_prices[0]
	orders.append(order)

	for curr_price in future_prices: 

		if curr_price < best_decision["left_price"]:
			best_decision = best_decision["left_decision"]
			if best_decision == None:
				break

			order = Order()
			order.dir = best_decision["dir"]
			order.amount = best_decision["amount"] * weight
			order.open_price = curr_price
			orders.append(order)



		if curr_price > best_decision["right_price"]:
			best_decision = best_decision["right_decision"]
			if best_decision == None:
				break

			order = Order()
			order.dir = best_decision["dir"]
			order.amount = best_decision["amount"] * weight
			order.open_price = curr_price
			orders.append(order)


	float_pnl = 0
	for order in orders:
		if (curr_price > order.open_price) == order.dir:
			float_pnl += abs(curr_price - order.open_price) * order.amount 
		else:
			float_pnl -= abs(curr_price - order.open_price) * order.amount 

	return float_pnl

def get_today_prediction(pair, timestamp, cnx):

	print (pair)

	if pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	global_prices, times, volumes = load_time_series(pair, None, False)

	if timestamp is not None:
		start_time = [i for i, t in enumerate(times) if t >= timestamp][0] + 24
	else:
		start_time = 365 * 24

	print ("start time", start_time, len(global_prices), timestamp)

	equity = 0
	orders = []
	pnls = []
	pos_closes = 0
	neg_closes = 0

	equity_factor = 1.0
	max_equity = 0
	opposite_dir = 	None
	day_count = 0
	for time in range(start_time, len(global_prices), 24):
		print (pair)

		prices = global_prices[max(0, time - 24 * 250): time]
		time_final = times[max(0, time - 24 * 250): time][-1]

		lines = []


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

		for percenitle in range(5, 80, 5):
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

	
		cursor = cnx.cursor()
		query = ("INSERT INTO support_resistance_historical_stats (pair, timestamp, price, json) values( \
			'" + pair + "', \
			'" + str(time_final) + "', \
			'" + str(prices[-1]) + "', \
			'" + str(json.dumps(reward_risk_ratio_map)) + "' \
			)")

		cursor.execute(query)
		cnx.commit()
		cursor.close()


while True:
	cnx = mysql.connector.connect(user='andstv48', password='Password81',
								  host='mysql.newscaptial.com',
								  database='newscapital')

	cursor = cnx.cursor()
	query = ("SELECT max(timestamp) FROM support_resistance_historical_stats where \
						pair = '" + sys.argv[1] + "' \
						")

	cursor.execute(query)

	setup_rows = []
	for row1 in cursor:
		setup_rows.append(row1)

	cursor.close()


	if setup_rows[0][0] != None:
		curr_time = setup_rows[0][0]
	else:
		curr_time = None


	try:
		get_today_prediction(sys.argv[1], curr_time, cnx)
	except:
		print ("conn error")
		pass

