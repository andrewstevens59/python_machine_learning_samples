


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
import paramiko
import re

import time
import datetime
import calendar
from dateutil import tz
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

from sklearn.linear_model import LinearRegression
from maximize_sharpe import *

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import json

import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import grid_delta as grid_delta
import markov_process as markov_process
from uuid import getnode as get_mac
import logging
import socket

from sklearn.ensemble import GradientBoostingRegressor
import download_calendar as download_calendar


class Order:

    def __init__(self):
        self.pair = ""
        self.dir = 0
        self.open_price = 0
        self.time = 0
        self.readable_time = ""
        self.amount = 0
        self.id = 0
        self.side = 0
        self.pnl = 0
        self.open_predict = 0
        self.tp_price = 0
        self.sl_price = 0
        self.hold_time = 0
        self.is_invert = False
        self.invert_num = 0
        self.reduce_amount = 0
        self.match_amount = 0
        self.equity_factor = 0
        self.carry_cost = 0




def get_calendar_df(pair, year): 

	if pair != None:
		currencies = [pair[0:3], pair[4:7]]
	else:
		currencies = None

	with open("/Users/callummc/Documents/economic_calendar/calendar_" + str(year) + ".txt") as f:
		content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	lines = [x.strip() for x in content] 

	from_zone = tz.gettz('America/New_York')
	to_zone = tz.gettz('UTC')

	contents = []

	for line in lines:
		line = line[len("2018-12-23 22:44:55 "):]
		toks = line.split(",")

		if currencies == None or toks[2] in currencies:

			est = datetime.datetime.strptime(toks[0] + " " + toks[1], "%b%d.%Y %I:%M%p")
			est = est.replace(tzinfo=from_zone)
			utc = est.astimezone(to_zone)

			time = calendar.timegm(utc.timetuple())

			non_decimal = re.compile(r'[^\d.]+')

			try:
				toks[4] = float(non_decimal.sub('', toks[4]))
				toks[5] = float(non_decimal.sub('', toks[5]))
				toks[6] = float(non_decimal.sub('', toks[6]))

				contents.append([toks[2], time, toks[3], toks[4], toks[5], toks[6]])
			except:
				pass

	return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous"])


def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(5000) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=Amer")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	rates = []
	prices = []
	labels = []
	times = []
	index = 0

	X = []
	y = []

	balance_map = {}
	while index < len(j):
		item = j[index]

		s = item['time']
		s = s[0 : s.index('.')]

		timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

		times.append(timestamp)
		prices.append(item['closeMid'])
		index += 1

	return prices, times


def load_time_series(symbol, year):

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir('/Users/callummc/') if isfile(join('/Users/callummc/', f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Candlestick_1_Hour_BID' in file:
			break

	with open('/Users/callummc/' + file) as f:
	    content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	from_zone = tz.tzlocal()
	to_zone = tz.tzutc()


	rates = []
	prices = []
	labels = []
	times = []
	price_range = []


	content = content[1:]

	if year != None:
		start_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".1.1 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())
		end_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".12.31 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())

	for index in range(len(content)):

		toks = content[index].split(',')

		s = toks[0][:-len(".000 GMT-0500")]
		local = datetime.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")

		local = local.replace(tzinfo=from_zone)

		# Convert time zone
		utc = local.astimezone(to_zone)

		time = calendar.timegm(utc.timetuple())

		if year == None or (time >= start_time and time < end_time):

			high = float(toks[2])
			low = float(toks[3])
			o_price = float(toks[1])
			c_price = float(toks[4])
			volume = float(toks[5])

			if high != low:
				rates.append([high - low, c_price - o_price])
				prices.append(c_price)
				price_range.append(c_price - o_price)
				times.append(time)

	return prices, times

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


	
def create_training_set(currency1, currency2, price_df, pair, year, test_calendar, description_map, feature_importances):

	X_train = []
	y_train = []

	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		global_feature_vector = []

		back_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).values.tolist()

		std = (price_df[price_df['times'] <= row['time']])['prices'].tail(24 * 5 * 4).diff(periods=4).std()
		mean = (price_df[price_df['times'] <= row['time']])['prices'].tail(24 * 5 * 4).diff(periods=4).mean()
		z_val = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).diff(periods=4).values.tolist()[-1]
		z_val = (z_val - mean) / std

		for i in [8]:
			before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * i).values.tolist()[0]
			calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= row["time"])]

			future_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24 * 5 * i).values.tolist()

			description_vector1_curr1 = [0] * len(description_map)
			description_vector2_curr1 = [0] * len(description_map)

			description_vector1_curr2 = [0] * len(description_map)
			description_vector2_curr2 = [0] * len(description_map)

			for index, history in calendar_history.iterrows():
				
				if history["currency"] == pair[0:3]:
					description_vector1_curr1[description_map[history["description"]]] = (history['actual'] - history['forecast'])
					description_vector2_curr1[description_map[history["description"]]] = (history['actual'] - history['previous'])
				else:
					description_vector1_curr2[description_map[history["description"]]] = (history['actual'] - history['forecast'])
					description_vector2_curr2[description_map[history["description"]]] = (history['actual'] - history['previous'])

			global_feature_vector += description_vector1_curr1 + description_vector2_curr1 + description_vector1_curr2 + description_vector2_curr2
		
		if len(feature_importances) > 0:
			reduced_feature_vector = []
			for feature, importance in zip(global_feature_vector, feature_importances):
				if importance > 1e-3:
					reduced_feature_vector.append(feature)

			global_feature_vector = reduced_feature_vector

		if len(future_prices) < 24 * 5 * 2:
			continue
		
		X_train.append(global_feature_vector)
		y_train.append(future_prices[-1] - future_prices[0])

	return X_train, y_train

def calculate_interest_cost(order, prices_df, time_stamp):

	open_years = (time_stamp - order.open_time) / (60 * 60 * 24 * 365)

	if order.dir:
		amount1 = order.amount * order.base_interest * open_years * order.base_rate
		amount2 = order.amount * order.quote_interest * open_years * order.quote_rate

	else:
		amount1 = order.amount * quote_interest * open_years * order.quote_rate
		amount2 = order.amount * base_interest * open_years * order.base_rate

	return amount1 - amount2

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

    return p_var

def min_volatility(df, pairs, global_std_map, optimization_bounds = (0.0, 1.0)):

	temp_df = pd.DataFrame()
	for pair in pairs:
		temp_df[pair] = df[pair].map(lambda x : x / global_std_map[pair])

	mean_returns = temp_df.mean()
	cov_matrix = temp_df.cov()

	num_assets = len(mean_returns)
	avg_weight = 1.0 / num_assets

	# constraints = [{'type': 'ineq', 'fun': lambda x: +x[i] - avg_weight * max_exposure} for i in range(num_assets)]

	constraints = []
	constraints.append({'type': 'eq', 'fun': lambda x: np.sum([abs(x1) for x1 in x]) - 1})

	from scipy.optimize import minimize
	args = (mean_returns, cov_matrix, 0)
	bound = optimization_bounds
	bounds = tuple(bound for asset in range(num_assets))
	result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
	                method='SLSQP', bounds=bounds, constraints=constraints)

	weights = result['x'].tolist() 
	return [w / result['fun'] for w in weights]

def back_test(currency1, currency2, price_df, pair, pairs, year, test_calendar, model_mean, description_map, feature_importances, std_map, global_std_map):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	print "no"

	equity = 0
	equity_curve = []
	orders = []
	returns = []

	end_of_year_time = test_calendar["time"].tail(1).values.tolist()[0]

	exposure_map = {}
	profit_map = {}
	orders_map = {}
	equity_curve_map = {}

	norm_map = {}

	anchor_price = None
	prev_time = None
	for index, row in test_calendar.iterrows():

		if row["time"] == prev_time:
			continue

		prev_time = row["time"]

		#print price_df[price_df['times'] >= row['time']]

		future_times = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['times'].values.tolist()
		future_prices = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['prices'].values.tolist()
		future_mult_factor = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['mult_factor'].values.tolist()

		back_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).values.tolist()

		future_prices = future_prices[12:]
		future_times = future_times[12:]
		future_mult_factors = future_mult_factor[12:]

		global_feature_vector = []

		for i in [8]:
			before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * i).values.tolist()[0]
			calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= row["time"])]

			description_vector1_curr1 = [0] * len(description_map)
			description_vector2_curr1 = [0] * len(description_map)

			description_vector1_curr2 = [0] * len(description_map)
			description_vector2_curr2 = [0] * len(description_map)

			for index, history in calendar_history.iterrows():
				
				if history["currency"] == pair[0:3]:
					description_vector1_curr1[description_map[history["description"]]] = (history['actual'] - history['forecast'])
					description_vector2_curr1[description_map[history["description"]]] = (history['actual'] - history['previous'])
				else:
					description_vector1_curr2[description_map[history["description"]]] = (history['actual'] - history['forecast'])
					description_vector2_curr2[description_map[history["description"]]] = (history['actual'] - history['previous'])

			global_feature_vector += description_vector1_curr1 + description_vector2_curr1 + description_vector1_curr2 + description_vector2_curr2
			

		if len(feature_importances) > 0:
			reduced_feature_vector = []
			for feature, importance in zip(global_feature_vector, feature_importances):
				if importance > 1e-3:
					reduced_feature_vector.append(feature)

			global_feature_vector = reduced_feature_vector

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]
		first_time = (price_df[price_df['times'] >= row['time']])['times'].head(1).values.tolist()[0]

		if anchor_price == None:
			anchor_price = first_price

		anchor_price = first_price
		anchor_time = first_time

		mean_prediction = model_mean.predict([global_feature_vector])[0]

		max_total_pnl = equity
		max_delta = 0
		found_news_release = False
		time_step = 11

		equity = 0
		curr_calendar_time = row["time"]

		#future_price_df = price_df[(price_df['times'] >= row['time'])]

		prev_mult_factor = None
		for future_price, future_time, future_mult_factor in zip(future_prices, future_times, future_mult_factors):
			time_step += 1

			if math.isnan(future_mult_factor):
				future_mult_factor = prev_mult_factor
			else:
				prev_mult_factor = future_mult_factor

			prev_equity = equity
			max_pnl = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					pnl = (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
				else:
					pnl = (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor

				max_pnl = max(max_pnl, pnl) 

				if future_time not in exposure_map:
					exposure_map[future_time] = order.amount
				else:
					exposure_map[future_time] += order.amount

				time_lapse = float(future_time - order.open_time)
				num_weeks = int(time_lapse / (60 * 60 * 24 * 7)) * (60 * 60 * 24 * 2)
				time_lapse -= num_weeks
				order.carry_cost = 150 * float((time_lapse / (60 * 60)) / 8760) * (float(order.amount) / 10000)
				order.pnl = pnl - order.carry_cost

			total_pnl = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					total_pnl += (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
				else:
					total_pnl += (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor

				total_pnl -= order.carry_cost

			if future_time not in equity_curve_map:
				equity_curve_map[future_time] = equity + total_pnl
			else:
				equity_curve_map[future_time] += equity + total_pnl

			max_total_pnl = max(max_total_pnl, equity + total_pnl)

			curr_pnl = equity + total_pnl

			delta = ((future_price - first_price) - mean_prediction) / 1
			#print abs(delta) / (time_step * 0.1)

			if abs(future_price - anchor_price) > max_delta * 2:

				if future_time not in std_map:
					std_map[future_time] = min_volatility((price_df[price_df['times'] <= future_time])[pairs].tail(24 * 5 * 4).diff(periods=24), pairs, global_std_map, optimization_bounds = (-1.0, 1.0))

				std = std_map[future_time][pairs.index(pair)]

				new_order = Order()
				new_order.open_price = future_price
				new_order.open_time = future_time
				new_order.base_interest = future_price
				new_order.dir = (delta) < 0
				new_order.amount = (abs(delta) * abs(std)) / (time_step * 0.1 * global_std_map[pair])

				'''
				if new_order.amount < 0.006:
					new_order.amount = 0
				'''

				orders.append(new_order)
				max_delta = abs(future_price - anchor_price)

			total_pnl = 0
			order_count = 0
			new_orders = []

			buy_count = 0
			sell_count = 0
			for order in orders:

				if len(orders) < 8 or order_count >= 1:
					new_orders.append(order)
				else:
					anchor_price = order.open_price
					max_delta = abs(future_price - anchor_price)
					if (future_price > order.open_price) == (order.dir):
						equity += (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
					else:
						equity += (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
					continue

				if order.dir:
					buy_count += 1
				else:
					sell_count += 1

				if (future_price > order.open_price) == (order.dir):
					total_pnl += (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
				else:
					total_pnl += (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor

				total_pnl -= order.carry_cost

				order_count += 1

			orders = new_orders

			if future_time not in profit_map:
				profit_map[future_time] = equity + total_pnl
				orders_map[future_time] = orders
			else:
				profit_map[future_time] += equity + total_pnl
				orders_map[future_time] += orders


			if (abs(future_price - anchor_price) < max_delta / (time_step * 0.1)):
				equity += total_pnl
				total_pnl = 0
				max_total_pnl = 0
				max_delta = 0
				anchor_price = first_price

				for order in orders:

					if (future_price > order.open_price) == (order.dir):
						returns.append(abs(future_price - order.open_price) * order.amount * future_mult_factor)
					else:
						returns.append(-abs(future_price - order.open_price) * order.amount * future_mult_factor)

				orders = []

				#break

		if found_news_release == False or True:
			anchor_price = None

			equity += total_pnl
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					returns.append(abs(future_price - order.open_price) * order.amount * future_mult_factor)
				else:
					returns.append(-abs(future_price - order.open_price) * order.amount * future_mult_factor)

			orders = []

	import collections
	#print collections.OrderedDict(sorted(exposure_map.items()))
	#print collections.OrderedDict(sorted(profit_map.items()))

	sorted_map = collections.OrderedDict(sorted(profit_map.items()))

	equity = 0

	order_amounts = []

	for v in sorted_map:
		equity_curve.append(equity)

		curr_order_amount = 0

		for order in orders_map[v]: 
			if order.open_time != v:
				curr_order_amount += order.amount

		total_amount = 0
		for order in orders_map[v]: 
			if order.open_time == v:

				'''
				if curr_order_amount > 0:
					order.pnl /= curr_order_amount
					order.amount /= curr_order_amount
				'''

				equity += order.pnl
			total_amount += order.amount

		order_amounts.append(total_amount)

		
		'''
		if equity < -4.0:
			break 
		'''

	return equity_curve, returns, np.mean(order_amounts), std_map, equity_curve_map


def align_prices(pairs, price_df):

	price_df.set_index('times', inplace=True)
	global_std_map = {}

	for pair in pairs:
		prices, times = load_time_series(pair, None)
		conv_df = pd.DataFrame()
		conv_df[pair] = prices
		conv_df['times'] = times

		std = conv_df[pair].diff(periods=24).std()
		global_std_map[pair] = std

		price_df = price_df.join(conv_df.set_index('times'))
	
	price_df.fillna(method='ffill', inplace=True)
	price_df.reset_index(inplace=True)

	return price_df, global_std_map

def create_data_set(year_range, currency_pair, select_year, description_map, feature_importances):

	X_train = []
	y_train = []

	X_test = []
	y_test = []

	prices, times = load_time_series(currency_pair, None)
	price_df = pd.DataFrame()
	price_df['prices'] = prices
	price_df['times'] = times

	
	for year in year_range:

		print "Year", year

		
		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 

			test_calendar = get_calendar_df(test_pair, year)

			x, y = create_training_set(first_currency, second_currency, price_df, test_pair, year, test_calendar, description_map, feature_importances)

			print len(x)
			if year != select_year:
				X_train += x
				y_train += y
			else:
				X_test += x
				y_test += y

	return X_train, y_train, X_test, y_test


#["AUD_CAD", "GBP_CAD", "NZD_CAD", "AUD_NZD"]

'''
from sklearn import metrics
import e



[fpr1, tpr1] = pickle.load(open("/tmp/roc_all_new", 'rb'))
[fpr2, tpr2] = pickle.load(open("/tmp/roc_all_old", 'rb'))

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'New Non Gyro Ground Truth AUC = %0.2f' % metrics.auc(fpr1, tpr1), color='r')
plt.plot(fpr2, tpr2, 'b', label = 'Current Gyro Ground Truth AUC = %0.2f' % metrics.auc(fpr2, tpr2), color='b')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''


'''
for year in range(2008, 2019):
	prices, times = load_time_series("NZD_USD", None)
	price_df = pd.DataFrame()
	price_df['prices'] = prices
	price_df['times'] = times
	price_df['mult_factor'] = 1.0

	print year

	X_interest_train, y_interest_train, X_interest_test, y_interest_test = interest_rate_model(price_df, "NZD_USD", year, interest_rate_df)

	if len(X_interest_test) == 0:
		continue

	

	print len(X_interest_train), len(X_interest_test)
	clf_interest = GradientBoostingRegressor(random_state=42)
	clf_interest.fit(X_interest_train, y_interest_train)
	print "interest score", clf_interest.score(X_interest_test, y_interest_test)
'''

std_map = {}
feature_importance_map = {}
for year in range(2010, 2019):
	print "Year ", year, "-----------------"

	total_pnl = 0
	pairs = ["USD_CAD", "USD_JPY", "EUR_JPY", "GBP_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY", "AUD_JPY"]
	#pairs = ["EUR_USD", "GBP_USD", "NZD_USD", "USD_CHF", "USD_CAD", "AUD_USD"]
	pairs = ["NZD_CAD", "AUD_NZD", "AUD_CAD"]
	#pairs = ["USD_CAD", "AUD_USD"]
	#pairs = currency_pairs
	#pairs = ['AUD_CAD', 'CHF_JPY', 'USD_CHF']

	#pairs = ['AUD_CAD', 'EUR_AUD', 'USD_CAD', 'CAD_CHF']
	#pairs = ['AUD_USD', 'EUR_GBP', 'GBP_AUD']

	pairs = ['USD_CAD', 'EUR_USD', 'CAD_CHF']

	equity_curve_maps = []
	for currency_pair in pairs:
	#for currency_pair in ["EUR_USD", "GBP_USD", "NZD_USD", "USD_CHF", "USD_CAD", "AUD_USD", "USD_JPY"]:
	#[EUR_AUD", "EUR_CAD", "EUR_GBP", "AUD_CAD", "GBP_CAD", "NZD_CAD", "AUD_NZD"]:
	#["EUR_JPY", "USD_JPY", "GBP_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY", "AUD_JPY"]:
	#["EUR_JPY", "USD_JPY", "EUR_AUD", "EUR_CAD", "EUR_GBP", "AUD_CAD", "GBP_CAD", "NZD_CAD", "AUD_NZD"]:
		print currency_pair

		description_map = {}
		for year1 in range(2008, 2019):
			descriptions = get_calendar_df(currency_pair, year1)["description"].values.tolist()
			for description in descriptions:
				if description not in description_map:
					description_map[description] = len(description_map)

		'''
		if currency_pair not in feature_importance_map:
			X_train1, y_train1, _, _ = create_data_set(range(2007, 2019), currency_pair, None, description_map, [])

			clf_mean = GradientBoostingRegressor(random_state=42)
			clf_mean.fit(X_train1, y_train1)
			feature_importances = clf_mean.feature_importances_
			feature_importance_map[currency_pair] = feature_importances
		else:
			feature_importances = feature_importance_map[currency_pair]
		'''

		feature_importances = []

		X_train, y_train, X_test, y_test = create_data_set(range(2008, 2019), currency_pair, year, description_map, feature_importances)
	
		#mean = np.mean(y_train)
		#y_train = [v - mean for v in y_train]

		#pickle.dump(X_train, open("/tmp/X_train", 'wb'))
		#pickle.dump(y_train, open("/tmp/y_train", 'wb'))
		#pickle.dump(kmeans, open("/tmp/kmeans", 'wb'))


		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 

			#X_train = pickle.load(open("/tmp/X_train", 'rb'))
			#y_train = pickle.load(open("/tmp/y_train", 'rb'))

			'''
			from sklearn.ensemble import GradientBoostingRegressor
			from sklearn.ensemble import RandomForestRegressor
			from sklearn.model_selection import RandomizedSearchCV
			import numpy as np
	 

			loss = ['ls', 'lad', 'huber']
			n_estimators = [100, 500, 900, 1100, 1500]
			max_depth = [2, 3, 5, 10, 15]
			min_samples_leaf = [1, 2, 4, 6, 8] 
			min_samples_split = [2, 4, 6, 10]
			max_features = ['auto', 'sqrt', 'log2', None]

			# Define the grid of hyperparameters to search
			hyperparameter_grid = {'loss': loss,
			    'n_estimators': n_estimators,
			    'max_depth': max_depth,
			    'min_samples_leaf': min_samples_leaf,
			    'min_samples_split': min_samples_split,
			    'max_features': max_features}

			clf_mean = RandomizedSearchCV(GradientBoostingRegressor(verbose=10, random_state=42), hyperparameter_grid, cv=5, scoring = 'neg_mean_absolute_error')

			clf_mean.fit(X_train, y_train)


			pickle.dump(clf, open("/tmp/news_model_" + currency_pair, 'wb'))

			pickle.dump(kmeans, open("/tmp/news_kmeans_" + currency_pair, 'wb'))

			continue
			'''


			clf_mean = GradientBoostingRegressor(random_state=42)
			clf_mean.fit(X_train, y_train)

			print "Train Score", clf_mean.score(X_train, y_train)
			print "Test Score", clf_mean.score(X_test, y_test)

			test_calendar = get_calendar_df(test_pair, year)

			prices, times = load_time_series(test_pair, None)
			price_df = pd.DataFrame()
			price_df['prices'] = prices
			price_df['times'] = times
			price_df['mult_factor'] = 1.0

			price_df, global_std_map = align_prices(pairs, price_df)
			
			'''
			if test_pair[4:7] == "USD":
				price_df['mult_factor'] = 1.0
			elif test_pair[4:7] + "_USD" in currency_pairs:
				prices, times = load_time_series(test_pair[4:7] + "_USD", None)

				conv_df = pd.DataFrame()
				conv_df['mult_factor'] = prices
				conv_df['times'] = times
				price_df = price_df.set_index('times').join(conv_df.set_index('times'))
				price_df.reset_index(inplace=True)
			else:
				prices, times = load_time_series("USD_" + test_pair[4:7], None)

				conv_df = pd.DataFrame()
				conv_df['mult_factor'] = [1.0 / p for p in prices]
				conv_df['times'] = times
				price_df = price_df.set_index('times').join(conv_df.set_index('times'))
				price_df.reset_index(inplace=True)
			'''

			equity_curve, returns, avg_amount, std_map, equity_curve_map = back_test(first_currency, second_currency, price_df, test_pair, pairs, year, test_calendar, clf_mean, description_map, feature_importances, std_map, global_std_map)
			if len(equity_curve) == 0:
				continue

			equity_curve_maps.append(equity_curve_map)

			if test_pair[4:7] != "JPY":
				total_pnl += equity_curve[-1] 
			else:
				total_pnl += equity_curve[-1] * 0.01
		

			print "Sharpe", np.mean(returns) / np.std(returns)
			print "Total Profit", total_pnl

			import datetime
			import numpy as np
			from matplotlib.backends.backend_pdf import PdfPages
			import matplotlib.pyplot as plt


			with PdfPages('/Users/callummc/Desktop/equity_curve_' + str(year) + '_' + test_pair + '.pdf') as pdf:
				plt.figure(figsize=(6, 4))
				plt.plot(equity_curve)

				pdf.savefig()  # saves the current figure into a pdf page
				plt.close() 

	min_profit = 0
	times = price_df["times"].values.tolist()
	for time in times:

		profit = 0
		for equity_curve_map in equity_curve_maps:

			if time in equity_curve_map:
				profit += equity_curve_map[time]

		min_profit = min(min_profit, profit)

	print "Min Profit", min_profit


		



