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

def train_model(setup_rows, pair):

	past_prices, past_times, volumes = load_time_series(pair, None, False)

	X = []
	y = []
	y_actual = []
	prices = []

	reward_risk_ratios_X = []
	for row in setup_rows:
		price = row[0]
		metadata = json.loads(row[1])
		timestamp = row[2]
		pair = row[3]

		keys = sorted(metadata.keys())
		reward_risk_ratios = []
		for key in keys:
			reward_risk_ratios.append(metadata[key])

		if len(reward_risk_ratios) != 15:
			continue

		reward_risk_ratios_X.append(reward_risk_ratios)
 
		if len(reward_risk_ratios_X) < 5:
			continue

		after_prices = [p for p, t in zip(past_prices, past_times) if t >= timestamp]
		if len(after_prices) < 24 * 5:
			continue

		flat_list = [item for sublist in reward_risk_ratios_X[-1:] for item in sublist]

		X.append(flat_list)
		y.append(after_prices[(24 * 5) - 1] - after_prices[0]) 
		prices.append(price)

	from sklearn.model_selection import cross_val_score
	from sklearn.metrics import roc_auc_score
	from sklearn.model_selection import train_test_split
	clf = xgb.XGBClassifier()

	if pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	profits = []
	for offset in range(0, len(y)-250):

		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
		X_train = X[:offset] + X[offset+250:]
		y_train = y[:offset] + y[offset+250:]

		X_test = X[offset + (22):offset+250]
		y_test = y[offset + (22):offset+250]

		y_train = [y1 > 0 for y1 in y_train]
		clf.fit(np.array(X_train), y_train)

		y_actual = y_test
		y_test = [y1 > 0 for y1 in y_test]
		print ("auc", roc_auc_score(y_test, clf.predict_proba(np.array(X_test))[:,1]))

		probs = clf.predict_proba(np.array(X_test))[:,1]

		returns = []
		for y1, p in zip(y_actual, probs):

			if (p - 0.5) < 0.2:
				continue

			if (p > 0.5) == (y1 > 0):
				returns.append((abs(y1) / pip_size) - 5)
			else:
				returns.append((-abs(y1) / pip_size) - 5)

		profits += returns
		print ("mean", np.mean(profits))

	preds = clf.predict_proba(np.array(X_test))[:,1]

	upper_threshold = np.percentile([abs(p - 0.5) for p in clf.predict_proba(np.array(X_train))[:,1]], 50)

	returns = []
	orders = []
	equity = 5000
	total_pnl = 0
	for index, y, prob, price in zip(range(len(y_actual)), y_actual, preds, price_test):

		new_orders = []
		total_pnl = 0
		for order in orders:

			if order.dir == (price > order.open_price):
				order.pnl = (abs(price - order.open_price) - (5 * pip_size)) * order.amount * 50
			else:
				order.pnl = (-abs(price - order.open_price) - (5 * pip_size)) * order.amount * 50

			'''
			if order.dir != (prob > 0.5):
				equity += order.pnl
				print ("opposite direction")
				continue
			'''

			if index - order.open_time > 5:
				equity += order.pnl
				print ("timeout")
				continue

			total_pnl += order.pnl
			new_orders.append(order)

		orders = new_orders


		if total_pnl > 0 and orders[-1].dir != (prob > 0.5):
			equity += total_pnl
			total_pnl = 0
			orders = []

		print ("Equity", equity + total_pnl)

		order = Order()
		order.dir = prob > 0.5
		order.open_price = price
		order.amount = abs(prob - 0.5)
		order.open_time = index

		orders.append(order)



	print (np.mean(returns), np.mean(returns) / np.std(returns))



def get_today_prediction(pair1, pair2):

	pair = "GBP_USD"

	'''
	cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

	#['AUD_USD', 'USD_JPY', 'GBP_USD', 'USD_CAD', 'EUR_USD']
	cursor = cnx.cursor()
	query = ("SELECT price, json, timestamp, pair FROM support_resistance_historical_stats  \
						where pair in ('{}') order by timestamp \
						".format(pair))

	print (query)
	cursor.execute(query)

	setup_rows = []
	for row1 in cursor:
		setup_rows.append(row1)

	cursor.close()

	pickle.dump(setup_rows, open("setup_rows_{}.pickle".format(pair), "wb"))
	'''
	
	setup_rows = pickle.load(open("setup_rows_{}.pickle".format(pair), "rb"))

	prev_revert = None
	past_prices = {}
	past_times = {}
	max_time = {}
	min_time = {}

	for select_pair in [pair]:
		past_prices1, past_times1, volumes = load_time_series(select_pair, None, False)
		past_prices[select_pair] = past_prices1
		past_times[select_pair] = past_times1
		max_time[select_pair] = 0
		min_time[select_pair] = 999999999999999

	for row1 in setup_rows:
		max_time[row1[3]] = max(max_time[row1[3]], row1[2])
		min_time[row1[3]] = min(min_time[row1[3]], row1[2])

	for key in max_time:
		print (key, min_time[key], max_time[key])


	start_time = max(min_time.values())
	end_time = min(max_time.values())

	
	

	for train_offset in range(240 + 60, len(setup_rows), 60):

		orders = []
		global_orders = []
		pnls = []
		equity = 5000
		day_count = 0
		equity_curve = []
		max_equity = 5000
		max_float = 5000
		leverages = []
		timestamps = []
		prev_reward_risk = None

		for row in setup_rows[train_offset:]:
			price = row[0]
			metadata = json.loads(row[1])
			timestamp = row[2]
			pair = row[3]

			if timestamp < start_time:
				continue

			if timestamp > end_time:
				break

			print (pair)

			if pair[4:7] == "JPY":
				pip_size = 0.01
			else:
				pip_size = 0.0001

			before_prices = [p for p, t in zip(past_prices[pair], past_times[pair]) if t <= timestamp]
			orders = [order for order in global_orders if order.pair == pair]

			if len(global_orders) > 0:
				total_pnl = sum([order.pnl for order in global_orders])
			else:
				total_pnl = 0

			keys = sorted(metadata.keys())

			reward_risk_ratios = []
			for key in keys:
				reward_risk_ratios.append(metadata[key])

			reward_risk_ratio = np.mean(reward_risk_ratios)
			print (reward_risk_ratio)

			total_amount = 0
			new_orders = []

			pos_orders = 0
			neg_orders = 0

			pos_pnl = 0
			neg_pnl = 0
			count = 0

			sell_pnl = 0
			buy_pnl = 0
			total_order_amount = 0
			for order in orders:

				if order.dir:
					total_amount += order.amount
				else:
					total_amount -= order.amount

				if order.dir == (price > order.open_price):
					order.pnl = (abs(price - order.open_price) - (5 * pip_size)) * order.amount * 100
				else:
					order.pnl = (-abs(price - order.open_price) - (5 * pip_size)) * order.amount * 100

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

			orders = new_orders

			mean = np.mean(before_prices[-24*20:])
			std = np.std(before_prices[-24*20:])
			z_score = (price - mean) / std

			if total_order_amount > 50 * (equity + total_pnl):
				print ("margin call")
				sys.exit(0)
			
			if len(orders) > 0:#(abs(neg_pnl) > 0 and abs(pos_pnl) > 0):
				new_orders = []
				loss_count = 0

				offset = int(len(orders) / 10)
				upper_percentile = np.percentile([order.pnl for order in orders], 80 + offset)
				lower_percentile = np.percentile([order.pnl for order in orders], 20 - offset)
				for order in orders:

			
					if abs(upper_percentile) > abs(lower_percentile):

			
						if order.pnl > upper_percentile and order.reverse < 2 and order.reverse >= 0:


							new = Order()
							new.dir = order.dir 
							new.amount = ((max_equity) / 5000) * math.log(len(orders))
							new.open_price = price
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
							new.amount = order.amount * 0.5* math.log(len(orders))

							if new.amount > 1: 
								new.open_price = price
								new.open_time = day_count
								new.reverse = order.reverse + 1
								new.pnl = 0
								new.pair = pair
								order.reverse = -1
								order.open_time = day_count
								new_orders.append(new)

						draw_down = ((equity + total_pnl) / max_equity)
		
						if (total_order_amount > 10 * (equity + total_pnl) or len(orders) > 6 and order.pnl <= lower_percentile) and loss_count == 0:
							equity += order.pnl
							total_pnl -= order.pnl
							loss_count = 1
							continue

					'''
					if len(orders) > 1 and order.pnl <= lower_percentile and loss_count == 0: 
						equity += order.pnl
						total_pnl -= order.pnl
						loss_count = 1
						continue
					'''

					'''
					if abs(pos_pnl) / max(1, abs(neg_pnl)) > 10 / len(orders): 
						equity += order.pnl
						total_pnl -= order.pnl
						continue
					'''

					new_orders.append(order) 

				orders = new_orders
			
			global_orders = [order for order in global_orders if order.pair != pair] + orders

			max_equity = max(max_equity, equity)
			max_float = max(max_float, equity + total_pnl)


			max_equity = min(max_equity, equity * 1.2)

			if ((total_order_amount > 10 * (equity + total_pnl) and equity >= max_equity)) or ((equity >= max_equity) and total_pnl > 0):

				max_equity = equity + total_pnl
				if len(global_orders) > 0:
					lower_percentile = np.percentile([order.pnl for order in global_orders], 10)

					ratio = max_float / (equity + total_pnl)

					if lower_percentile < 0:

						new_orders = []
						for order in global_orders:

							if True:# or order.pnl > lower_percentile: 
								equity += order.pnl
								total_pnl -= order.pnl
								continue

							new_orders.append(order)

						global_orders = new_orders
						orders = [order for order in orders if order in global_orders]
						

			if abs(reward_risk_ratio) > 0 and (np.percentile(reward_risk_ratios, 30) > 0) != (np.percentile(reward_risk_ratios, 70) > 0):

				if len(orders) > 0:
					order = Order()


					order.dir = price > orders[-1].open_price


					if True:#order.dir == (z_score > 0) :
						pip_diff = abs(price - orders[-1].open_price) / pip_size
						order.amount = ((max_equity) / 5000) * len(orders)
				 		order.open_price = price
						order.open_time = day_count
						order.reverse = 0
						order.pair = pair
						order.pnl = 0

						orders.append(order)
				else:

					order = Order()
					order.dir = reward_risk_ratio > 0
					
					if True:#order.dir == (z_score > 0):
						order.amount = 1 * ((max_equity) / 5000)  * abs(z_score)
						order.open_price = price
						order.open_time = day_count
						order.reverse = 0
						order.pair = pair
						order.pnl = 0
						orders.append(order)
						prev_reward_risk = reward_risk_ratio > 0

			pnls.append(total_pnl)
			equity_curve.append(equity + total_pnl)
			leverages.append(float(total_order_amount) / (equity + total_pnl))
			timestamps.append(timestamp)

			global_orders = [order for order in global_orders if order.pair != pair] + orders

			print ("max_equity", max_equity, "equity", equity + total_pnl, total_pnl, len(orders), abs(reward_risk_ratio) )
			print ("day count", day_count)
			day_count += 1

		returns = [a - b for a, b in zip(equity_curve[1:], equity_curve[:-1])]
		sharpe = np.mean(returns) / np.std(returns)
		print ("Sharpe", sharpe, max_equity)

		from datetime import datetime

		with PdfPages('trend_strategy_{}.pdf'.format(train_offset)) as pdf:
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
   

