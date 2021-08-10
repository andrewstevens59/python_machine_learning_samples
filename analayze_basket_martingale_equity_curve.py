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
import json

import re

import matplotlib

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

def get_today_prediction(pair1, pair2):



	prev_revert = None
	past_prices = {}
	past_times = {}
	max_time = {}
	min_time = {}

	select_pairs = []
	for select_pair in currency_pairs:
		if "AUD" in select_pair or "CAD" in select_pair: 
			past_prices1, past_times1, volumes = load_time_series(select_pair, None, False)
			past_prices[select_pair] = past_prices1
			past_times[select_pair] = past_times1
			max_time[select_pair] = 0
			min_time[select_pair] = 999999999999999
			select_pairs.append(select_pair)

	start_time = max(min_time.values())
	end_time = min(max_time.values())

	orders = []
	pnls = []
	equity = 5000
	day_count = 0
	equity_curve = []
	max_equity = 5000
	max_float = 5000
	leverages = []
	timestamps = []

	for timestamp in past_times["AUD_CAD"][-365 * 24:]:
		print "here"

		price_deltas = []
		prices = {}
		for pair in select_pairs:

			if pair[4:7] == "JPY":
				pip_size = 0.01
			else:
				pip_size = 0.0001

			before_prices = [p for p, t in zip(past_prices[pair], past_times[pair]) if t <= timestamp]
			price_delta = before_prices[-1] - before_prices[max(0, -24 * 5)]
			price_delta /= pip_size

			price_deltas.append([pair, price_delta])
			prices[pair] = before_prices[-1]


		price_deltas = sorted(price_deltas, key=lambda x: x[1])
		print (price_deltas)

		total_amount = 0
		new_orders = []

		pos_orders = 0
		neg_orders = 0

		pos_pnl = 0
		neg_pnl = 0
		count = 0

		sell_pnl = 0
		buy_pnl = 0
		total_pnl = 0
		total_order_amount = 0
		for order in orders:

			if order.dir:
				total_amount += order.amount
			else:
				total_amount -= order.amount

			if order.dir == (prices[order.pair] > order.open_price):
				order.pnl = (abs(prices[order.pair] - order.open_price) - (5 * pip_size)) * order.amount * 500
			else:
				order.pnl = (-abs(prices[order.pair] - order.open_price) - (5 * pip_size)) * order.amount * 500

			if pair[4:7] == "JPY":
				order.pnl *= 0.01

			if order.pnl > 0:
				pos_orders += order.amount
				pos_pnl += order.pnl
			else:
				neg_orders += order.amount
				neg_pnl += order.pnl

			if order.dir:
				buy_pnl += order.pnl
			else:
				sell_pnl += order.pnl

			new_orders.append(order)
			total_order_amount += order.amount
			total_pnl += order.pnl

		orders = new_orders

		if False and len(orders) > 0:#(abs(neg_pnl) > 0 and abs(pos_pnl) > 0):
			new_orders = []
			loss_count = 0

			upper_percentile = np.percentile([order.pnl for order in orders], 80)
			lower_percentile = np.percentile([order.pnl for order in orders], 20)
			for order in orders:

		
				if abs(upper_percentile) > abs(lower_percentile):

		
					if order.pnl > upper_percentile and order.reverse < 2 and order.reverse >= 0:


						new = Order()
						new.dir = order.dir 
						new.amount = ((max_equity) / 5000) * len(orders)* (order.reverse + 1)
						new.open_price = prices[order.pair]
						new.open_time = day_count
						new.reverse = order.reverse + 1
						new.pnl = 0
						new.pair = pair
						order.reverse = -1
						order.open_time = day_count
						new_orders.append(new)

	
					if len(orders) > 6 and order.pnl >= upper_percentile and loss_count == 0: 
						equity += order.pnl
						total_pnl -= order.pnl
						loss_count = 1
						continue
				else:

					if  order.pnl < lower_percentile and order.reverse >= 0:
					
						new = Order()
						new.dir = (order.dir == False)
						new.amount = ((max_equity) / 5000) * len(orders)* (order.reverse + 1)
						new.open_price = prices[order.pair]
						new.open_time = day_count
						new.reverse = order.reverse + 1
						new.pnl = 0
						new.pair = pair
						order.reverse = -1
						order.open_time = day_count
						new_orders.append(new)
	
					if len(orders) > 6 and order.pnl <= lower_percentile and loss_count == 0:
						equity += order.pnl
						total_pnl -= order.pnl
						loss_count = 1
						continue


				new_orders.append(order) 

			orders = new_orders
		
		if ((equity + total_pnl) / max_equity > 1.01):

			new_orders = []
			for order in orders:

				if True:#total_pnl / max_equity > 1.01 or order.pair not in [price_deltas[-1][0], price_deltas[0][0]]:# or order.pnl > lower_percentile: 
					equity += order.pnl
					total_pnl -= order.pnl
					continue

				new_orders.append(order)

			orders = new_orders
			max_equity = equity
			print ("close")
					
		order = Order()
		order.dir = False
		order.amount = 1 * ((max_equity) / 5000)  
		order.open_price = prices[price_deltas[0][0]]
		order.open_time = day_count
		order.reverse = 0
		order.pair = price_deltas[0][0]
		order.pnl = 0
		orders.append(order)

		order = Order()
		order.dir = True
		order.amount = 1 * ((max_equity) / 5000) 
		order.open_price = prices[price_deltas[-1][0]]
		order.open_time = day_count
		order.reverse = 0
		order.pair = price_deltas[-1][0]
		order.pnl = 0
		orders.append(order)

		print ([price_deltas[-1][0], price_deltas[0][0]])



		pnls.append(total_pnl)
		equity_curve.append(equity + total_pnl)
		leverages.append(float(total_order_amount) / (equity + total_pnl))
		timestamps.append(timestamp)

		print ("max_equity", max_equity, "equity", equity + total_pnl, total_pnl, len(orders))
		print ("day count", day_count)
		day_count += 1

	returns = [a - b for a, b in zip(equity_curve[1:], equity_curve[:-1])]
	sharpe = np.mean(returns) / np.std(returns)
	print ("Sharpe", sharpe, max_equity)

	from datetime import datetime

	with PdfPages('trend_strategy.pdf') as pdf:
		plt.title("Equity Curve Backtest {} / {}".format(pair1, pair2))
		plt.plot(range(len(equity_curve)), equity_curve, color='b', label="Sharpe {}".format(sharpe * math.sqrt(250)))
		plt.ylabel('Equity')
		plt.xlabel('Year')
		plt.legend()

		x_tick_indexes = [index for index in range(0, len(timestamps), 500)]
		plt.xticks(x_tick_indexes, [datetime.utcfromtimestamp(timestamps[t]).strftime('%Y-%m') for t in x_tick_indexes], rotation=30)

		if sharpe > 0.01:
			plt.savefig("backtest_plots/{}_{}_backtest.png".format(pair1, pair2))

		pdf.savefig()
		plt.close()

get_today_prediction("NZD_USD", "EUR_NZD")
pickle.dump({}, open("backtest_plots/{}_{}_processed.pickle".format(pair1, pair2), "wb"))
   
for index1 in range(len(currency_pairs)):
	for index2 in range(index1 + 1, len(currency_pairs)):

		pair1 = currency_pairs[index1]
		pair2 = currency_pairs[index2]

		if "USD" not in pair1:
			continue

		if "EUR" not in pair1 and "EUR" not in pair2:
			continue

		if pair1[0:3] not in pair2 and pair1[4:7] not in pair2:
			continue

		import os.path
		if os.path.isfile("backtest_plots/{}_{}_processed.pickle".format(pair1, pair2)):
			continue

		if os.path.isfile("backtest_plots/{}_{}_backtest.png".format(pair1, pair2)):
			continue

		get_today_prediction(pair1, pair2)
		pickle.dump({}, open("backtest_plots/{}_{}_processed.pickle".format(pair1, pair2), "wb"))
   

