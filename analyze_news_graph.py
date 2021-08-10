


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

'''
def load_time_series(symbol, year):

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir('/Users/callummc/') if isfile(join('/Users/callummc/', f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Ask' not in file:
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

		local = datetime.datetime.strptime(toks[0], "%Y.%m.%d %H:%M:%S")

		local = local.replace(tzinfo=from_zone)

		# Convert time zone
		utc = local.astimezone(to_zone)

		time = calendar.timegm(utc.timetuple())

		if year == None or (time >= start_time and time < end_time):

			high = float(toks[2])
			low = float(toks[3])
			o_price = float(toks[1])
			c_price = float(toks[4])

			rates.append([high - low, c_price - o_price])
			prices.append(c_price)
			price_range.append(c_price - o_price)

			times.append(time)

	return prices, times
'''

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

def interest_rate_model(price_df, pair, year, interest_rate_df):


	test_interest_df = interest_rate_df[(interest_rate_df["CURRENCY"] == pair[0:3]) | (interest_rate_df["CURRENCY"] == pair[4:7])]

	test_interest_df.sort_values(by=['DATE'], inplace=True)

	X_train = []
	y_train = []

	X_test = []
	y_test = []

	prev_interest_rates = [0] * 4
	interest_times = set()
	for index, row in test_interest_df.iterrows():

		feature_vector = [0] * 4


		if row["CURRENCY"] == pair[0:3]:
			feature_vector[0] = row["BID"] - prev_interest_rates[0]
			feature_vector[1] = row["ASK"] - prev_interest_rates[1]
			prev_interest_rates[0] = row["BID"]
			prev_interest_rates[1] = row["ASK"]
		else:
			feature_vector[2] = row["BID"] - prev_interest_rates[2]
			feature_vector[3] = row["ASK"] - prev_interest_rates[3]
			prev_interest_rates[2] = row["BID"]
			prev_interest_rates[3] = row["ASK"]

		future_prices = (price_df[price_df['times'] >= row['DATE']])['prices'].head(24 * 2).values.tolist()

		if len(future_prices) < 12:
			continue

		interest_times.add(row["DATE"])

		if row["YEAR"] == year:
			y_test.append(future_prices[-1] - future_prices[0])
			X_test.append(feature_vector)
		else:
			y_train.append(future_prices[-1] - future_prices[0])
			X_train.append(feature_vector)
		

	return X_train[1:], y_train[1:], X_test[1:], y_test[1:], interest_times




def create_training_set(currency1, currency2, price_df, pair, year, test_calendar, description_map, feature_importances):

	X_train = []
	y_train = []

	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		global_feature_vector = []

		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(24 * 5 * 2).values.tolist()

		back_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).values.tolist()

		std = (price_df[price_df['times'] <= row['time']])['prices'].tail(24 * 5 * 4).diff(periods=4).std()
		mean = (price_df[price_df['times'] <= row['time']])['prices'].tail(24 * 5 * 4).diff(periods=4).mean()
		z_val = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).diff(periods=4).values.tolist()[-1]
		z_val = (z_val - mean) / std

		if len(future_prices) < 12:
			continue

		deltas1 = []
		for index in range(1, 12):
			deltas1.append(future_prices[index] - future_prices[0])

		deltas2 = []
		for index in range(0, len(back_prices) - 1):
			deltas2.append(back_prices[-1] - back_prices[index])

		for i in [2, 4, 8]:
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
			
		global_feature_vector += deltas1 + deltas2 

		for delta_i in deltas1:
			for delta_j in deltas2:
				global_feature_vector += [delta_i - delta_j]
		
		if len(feature_importances) > 0:
			reduced_feature_vector = []
			for feature, importance in zip(global_feature_vector, feature_importances):
				if importance > 1e-3:
					reduced_feature_vector.append(feature)

			global_feature_vector = reduced_feature_vector
		
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

		if len(future_prices) < 12:
			continue

		deltas1 = []
		for index in range(1, 12):
			deltas1.append(future_prices[index] - future_prices[0])

		deltas2 = []
		for index in range(0, len(back_prices) - 1):
			deltas2.append(back_prices[-1] - back_prices[index])

		future_prices = future_prices[12:]
		future_times = future_times[12:]
		future_mult_factors = future_mult_factor[12:]

		global_feature_vector = []

		for i in [2, 4, 8]:
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
			
		global_feature_vector += deltas1 + deltas2 

		for delta_i in deltas1:
			for delta_j in deltas2:
				global_feature_vector += [delta_i - delta_j]

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
		print currency_pair

		description_map = {}
		for year1 in range(2008, 2019):
			descriptions = get_calendar_df(currency_pair, year1)["description"].values.tolist()
			for description in descriptions:
				if description not in description_map:
					description_map[description] = len(description_map)

		create_training_set(first_currency, second_currency, price_df, currency_pair, year, test_calendar, description_map, feature_importances)

	min_profit = 0
	times = price_df["times"].values.tolist()
	for time in times:

		profit = 0
		for equity_curve_map in equity_curve_maps:

			if time in equity_curve_map:
				profit += equity_curve_map[time]

		min_profit = min(min_profit, profit)

	print "Min Profit", min_profit


		



