


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

		if pair in file and 'Candlestick_1_Hour_ASK' in file:
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


def find_modal_points(currency1, currency2, price_df, pair, year, test_calendar):

	test_calendar = get_calendar_df(pair, year)

	price_deltas = []
	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()[12:48]

		if len((price_df[price_df['times'] >= row['time']])) == 0:
			continue

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]

		before_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(48).values.tolist()
		before_returns = [a - b for a, b in zip(before_prices[1:], before_prices[:-1])]
		before_mean = np.mean(before_returns)
		before_std = np.std(before_returns)


		for future_time, time_index in zip(future_times, range(len(future_times))):

			prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()
			times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()

			for price_v, time_v in zip(prices, times):

				if time_v <= future_time:

					price_deltas.append(price_v - first_price)

			break

	return price_deltas

def find_price_barrier_membership(prices, window_width):

	kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, 
		           random_state = 42).fit([[v] for v in prices[-window_width:]])

	cluster_ids = kmeans.predict([[v] for v in prices[-window_width:]])

	
	cluster_set = [[kmeans.cluster_centers_[cluster_id], cluster_id] for cluster_id in range(len(kmeans.cluster_centers_))]

	cluster_set = sorted(cluster_set, key=lambda x: x[0], reverse=False)
	

	feature_vector = [0] * 10
	for cluster_id in cluster_ids:

		for item_index in range(len(cluster_set)):
			if cluster_id == cluster_set[item_index][1]:
				feature_vector[item_index] += 1
				break
		

	return [float(v) / len(cluster_ids) for v in feature_vector]


def find_min_reversal_time(price_df, news_time):
	future_times = (price_df[price_df['times'] >= news_time])['times'].head(1000).values.tolist()
	future_prices = (price_df[price_df['times'] >= news_time])['prices'].head(1000).values.tolist()


	start_price = future_prices[0]

	is_above = False
	is_below = False
	count = 0

	above_count = 0
	below_count = 0
	for future_price in future_prices[12:]:

		if future_price > start_price:
			is_above = True
			above_count += 1

		if future_price < start_price:
			is_below = True
			below_count += 1

		if is_above and is_below:

			if below_count > above_count:
				return -count

			return count

		count += 1
		
	return 0


def create_training_set(currency1, currency2, price_df, pair, year, kmean_groups, test_calendar, description_map):

	

	X_train = []
	y_train_mean = []
	y_train_std = []

	reversal_X = []
	reveral_times = []

	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		before_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(48).values.tolist()
		before_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(48).values.tolist()
		before_returns = [a - b for a, b in zip(before_prices[1:], before_prices[:-1])]
		before_mean = np.mean(before_returns)
		before_std = np.std(before_returns)

		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(50).values.tolist()[12:48]
		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(50).values.tolist()[12:48]

		if len(future_prices) < 12:
			continue

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]
		before_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(48).values.tolist()
		before_returns = [a - b for a, b in zip(before_prices[1:], before_prices[:-1])]
		before_mean = np.mean(before_returns)
		before_std = np.std(before_returns)

		deltas = []
		for future_price in future_prices:
			deltas.append(future_price - first_price)

		y_train_mean.append(future_prices[-1] - first_price)
		y_train_std.append(np.std(deltas))

		reversal_time = find_min_reversal_time(price_df, row['time'])

		if abs(reversal_time) > 1:
			reveral_times.append(reversal_time)

		for future_time, time_index in zip(future_times, range(len(future_times))):

			prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()
			times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()

			global_featue_vector = []
			for kmeans in kmean_groups:

				count = 0
				feature_vector = [0] * len(kmeans.cluster_centers_)
				for price_v, time_v in zip(prices, times):

					if time_v <= future_time:
						cluster_id = kmeans.predict([[price_v - first_price]])[0]
						feature_vector[cluster_id] += 1
						count += 1

				feature_vector = [float(v) / count for v in feature_vector] 

				global_featue_vector += feature_vector 

			'''
			global_featue_vector += [row['actual'] - row['forecast'], row['actual'] - row['previous']]

			description_vector = [0] * len(description_map)
			description_vector[description_map[row["description"]]] = 1.0
			global_featue_vector += description_vector
			'''

			'''
			window_width = 24
			for range_index in range(4):
				global_featue_vector += find_price_barrier_membership(before_prices, window_width)
				window_width += 24
			'''

			if abs(reversal_time) > 1:
				reversal_X.append(global_featue_vector)

			X_train.append(global_featue_vector)
			break
		

	return X_train, y_train_mean, y_train_std, reversal_X, reveral_times

def back_test(currency1, currency2, price_df, pair, year, kmean_groups, test_calendar, model_mean, model_reversal, description_map):

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

	anchor_price = None
	prev_time = None
	for index, row in test_calendar.iterrows():

		if row["time"] == prev_time:
			continue

		prev_time = row["time"]

		#print price_df[price_df['times'] >= row['time']]

		future_times = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['times'].values.tolist()[12:]
		future_prices = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['prices'].values.tolist()[12:]

		if len(future_prices) < 12:
			continue

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]
		first_time = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]

		if anchor_price == None:
			anchor_price = first_price

		anchor_price = first_price
		anchor_time = first_time

		before_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(48).values.tolist()
		before_returns = [a - b for a, b in zip(before_prices[1:], before_prices[:-1])]
		before_mean = np.mean(before_returns)
		before_std = np.std(before_returns)

		for future_time, time_index in zip(future_times, range(len(future_times))):

			prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()
			times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()

			global_featue_vector = []
			for kmeans in kmean_groups:

				count = 0
				feature_vector = [0] * len(kmeans.cluster_centers_)
				for price_v, time_v in zip(prices, times):

					if time_v <= future_time:
						cluster_id = kmeans.predict([[price_v - first_price]])[0]
						feature_vector[cluster_id] += 1
						count += 1

				feature_vector = [float(v) / count for v in feature_vector] 

				global_featue_vector += feature_vector

			'''
			global_featue_vector += [row['actual'] - row['forecast'], row['actual'] - row['previous']]

			description_vector = [0] * len(description_map)
			description_vector[description_map[row["description"]]] = 1.0
			global_featue_vector += description_vector
			'''

			'''
			window_width = 24
			for range_index in range(4):
				global_featue_vector += find_price_barrier_membership(before_prices, window_width)
				window_width += 24
			'''
			

			mean_prediction = model_mean.predict([global_featue_vector])[0]
			break

		max_total_pnl = equity
		max_delta = 0
		found_news_release = False
		time_step = 11

		curr_calendar_time = row["time"]
		for future_price, future_time in zip(future_prices, future_times):
			time_step += 1
			
			#if len(test_calendar[(test_calendar["time"] <= future_time) & (test_calendar["time"] > curr_calendar_time)]) > 0:
		
			
			max_pnl = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					pnl = (abs(future_price - order.open_price) - commission) * order.amount
				else:
					pnl = (-abs(future_price - order.open_price) - commission) * order.amount

				max_pnl = max(max_pnl, pnl)

			total_pnl = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					total_pnl += (abs(future_price - order.open_price) - commission) * order.amount
				else:
					total_pnl += (-abs(future_price - order.open_price) - commission) * order.amount

			max_total_pnl = max(max_total_pnl, equity + total_pnl)

			if abs(future_price - anchor_price) > max_delta * 2:
				delta = ((future_price - first_price) - mean_prediction) / 1

				new_order = Order()
				new_order.open_price = future_price
				new_order.open_time = future_time
				new_order.dir = delta < 0
				new_order.amount = 1.0 / (time_step * 0.1)

				orders.append(new_order)
				max_delta = abs(future_price - anchor_price)

			total_pnl = 0
			order_count = 0
			new_orders = []
			for order in orders:

				if len(orders) < 8 or order_count >= 1:
					new_orders.append(order)
				else:
					anchor_price = order.open_price
					max_delta = abs(future_price - anchor_price)
					if (future_price > order.open_price) == (order.dir):
						equity += (abs(future_price - order.open_price) - commission) * order.amount
					else:
						equity += (-abs(future_price - order.open_price) - commission) * order.amount
					continue

				if (future_price > order.open_price) == (order.dir):
					total_pnl += (abs(future_price - order.open_price) - commission) * order.amount
				else:
					total_pnl += (-abs(future_price - order.open_price) - commission) * order.amount

				order_count += 1

			orders = new_orders

			if (abs(future_price - anchor_price) < max_delta / (time_step * 0.1)):
				equity += total_pnl
				total_pnl = 0
				max_total_pnl = 0
				max_delta = 0
				anchor_price = first_price

				for order in orders:

					if (future_price > order.open_price) == (order.dir):
						returns.append(abs(future_price - order.open_price) * order.amount)
					else:
						returns.append(-abs(future_price - order.open_price) * order.amount)

				orders = []


		if found_news_release == False or True:
			anchor_price = None

			equity += total_pnl
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					returns.append(abs(future_price - order.open_price) * order.amount)
				else:
					returns.append(-abs(future_price - order.open_price) * order.amount)

			orders = []

		equity_curve.append(equity)

	return equity_curve, returns


def create_data_set(year_range, currency_pair, select_year, description_map):
	price_trends = []
	for year in year_range:
		if year == select_year:
			continue

		print "Year", year
		
		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 


			print test_pair
			prices, times = load_time_series(test_pair, None)

			
			price_df = pd.DataFrame()
			price_df['prices'] = prices
			price_df['times'] = times

			test_calendar = get_calendar_df(test_pair, year)

			price_trends += find_modal_points(first_currency, second_currency, price_df, test_pair, year, test_calendar)
			
	kmean_groups = []
	kmean_sizes = [20, 40, 80]

	for kmean_size in kmean_sizes:

		kmeans = KMeans(n_clusters=kmean_size, init='k-means++', max_iter=100, n_init=1, 
		                random_state = 42).fit([[v] for v in price_trends])

		kmean_groups.append(kmeans)

	X_train = []
	y_train_mean = []
	y_train_std = []


	X_rev = []
	y_rev = []
	for year in year_range:
		if year == select_year:
			continue

		print "Year", year
		
		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 

			prices, times = load_time_series(test_pair, None)
			price_df = pd.DataFrame()
			price_df['prices'] = prices
			price_df['times'] = times

			test_calendar = get_calendar_df(test_pair, year)

			x, y_mean, y_std, rev_x, rev_y = create_training_set(first_currency, second_currency, price_df, test_pair, year, kmean_groups, test_calendar, description_map)

			X_train += x
			y_train_mean += y_mean
			y_train_std += y_std

			X_rev += rev_x
			y_rev += rev_y

	return X_train, y_train_mean, y_train_std, kmean_groups, X_rev, y_rev

#["AUD_CAD", "GBP_CAD", "NZD_CAD", "AUD_NZD", "GBP_NZD"]

'''
import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

o_df = pd.read_csv("/tmp/X_train.pickle")


usage_df = o_df[o_df["usage"] == 1].sample(1000)
no_usage_df = o_df[o_df["usage"] == 0].sample(1000)

usage = usage_df[["accel_x", "accel_y", "accel_z"]].values.tolist()
no_usage = no_usage_df[["accel_x", "accel_y", "accel_z"]].values.tolist()

X = usage + no_usage

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1,
                    random_state = 42).fit(X)

o_df = pd.read_csv("/tmp/X_test.pickle")
usage_cluster_ids = kmeans.predict(o_df[o_df["usage"] == 1][["accel_x", "accel_y", "accel_z"]].values.tolist())
no_usage_cluster_ids = kmeans.predict(o_df[o_df["usage"] == 0][["accel_x", "accel_y", "accel_z"]].values.tolist())
mat_use = np.zeros((5, 5))
mat_no_use = np.zeros((5, 5))

for index in range(len(usage_cluster_ids) - 1):
	mat_use[usage_cluster_ids[index]][usage_cluster_ids[index + 1]] += 1.0 / len(usage_cluster_ids)

for index in range(len(no_usage_cluster_ids) - 1):
	mat_no_use[no_usage_cluster_ids[index]][no_usage_cluster_ids[index + 1]] += 1.0 / len(no_usage_cluster_ids)

print mat_use
print mat_no_use

mat = np.zeros((5, 5))
for i in range(5):
	for j in range(5):
		if (mat_use[i][j] + mat_no_use[i][j]) > 0:
			mat[i][j] = float(mat_use[i][j] - mat_no_use[i][j]) / (mat_use[i][j] + mat_no_use[i][j])

print mat


usage_df["norm"] = usage_df.apply(lambda x: math.sqrt((x["accel_x"] * x["accel_x"]) + (x["accel_y"] * x["accel_y"]) + (x["accel_z"] * x["accel_z"])), axis=1)
no_usage_df["norm"] = no_usage_df.apply(lambda x: math.sqrt((x["accel_x"] * x["accel_x"]) + (x["accel_y"] * x["accel_y"]) + (x["accel_z"] * x["accel_z"])), axis=1)


with PdfPages('/Users/callummc/Desktop/APU_test.pdf') as pdf:

	for column in ["accel_x", "accel_y", "accel_z", "norm"]:

		
		x_accel1 = usage_df[[column]].values.tolist()
		x_accel1 = [x[0] for x in x_accel1]

		x_accel2 = no_usage_df[[column]].values.tolist()
		x_accel2 = [x[0] for x in x_accel2]

		plt.hist(x_accel1, bins=range(-20, 20), alpha=0.5)  # arguments are passed to np.histogram
		plt.hist(x_accel2, bins=range(-20, 20), alpha=0.5) 
		plt.title(column)

		pdf.savefig()  # saves the current figure into a pdf page
		plt.close() 

	plt.imshow(np.array(mat), cmap='hot', interpolation='nearest')
	plt.title("Heat Map (normalized) Lighter Color Means Higher Active Phone Use")
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close() 

sys.exit(0)
'''


for year in range(2008, 2018):

	print "Year ", year, "-----------------"

	total_pnl = 0
	for currency_pair in ["AUD_CAD", "NZD_CAD", "AUD_NZD"]:
		print currency_pair

		description_map = {}
		for year1 in range(2007, 2018):
			descriptions = get_calendar_df(currency_pair, year1)["description"].values.tolist()
			for description in descriptions:
				if description not in description_map:
					description_map[description] = len(description_map)

		X_train, y_train_mean, y_train_std, kmeans, X_rev, y_rev = create_data_set(range(year-1, year), currency_pair, year, description_map)

		print "revert", len(X_rev), len(y_rev)

		#pickle.dump(X_train, open("/tmp/X_train", 'wb'))
		#pickle.dump(y_train, open("/tmp/y_train", 'wb'))
		#pickle.dump(kmeans, open("/tmp/kmeans", 'wb'))


		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 

			#X_train = pickle.load(open("/tmp/X_train", 'rb'))
			#y_train = pickle.load(open("/tmp/y_train", 'rb'))

			from sklearn.ensemble import GradientBoostingRegressor
			from sklearn.ensemble import RandomForestRegressor
			from sklearn.model_selection import RandomizedSearchCV
			import numpy as np
	 
	 		'''
			clf = GradientBoostingRegressor(verbose=10, random_state=42)

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

			clf = RandomizedSearchCV(clf, hyperparameter_grid, cv=5, scoring = 'neg_mean_absolute_error')

			clf.fit(X_train, y_train_mean)
			pickle.dump(clf, open("/tmp/news_model_" + currency_pair, 'wb'))

			pickle.dump(kmeans, open("/tmp/news_kmeans_" + currency_pair, 'wb'))

			continue
			'''

			clf_mean = GradientBoostingRegressor(random_state=42)
			clf_mean.fit(X_train, y_train_mean)

			clf_std = GradientBoostingRegressor(random_state=42)
			clf_std.fit(X_train, y_train_std)

			rev_clf = GradientBoostingRegressor(random_state=42)
			rev_clf.fit(X_rev, y_rev)

			test_calendar = get_calendar_df(test_pair, year)

			prices, times = load_time_series(test_pair, None)

			price_df = pd.DataFrame()
			price_df['prices'] = prices
			price_df['times'] = times

					
			equity_curve, returns = back_test(first_currency, second_currency, price_df, test_pair, year, kmeans, test_calendar, clf_mean, rev_clf, description_map)

			if test_pair[4:7] != "JPY":
				if test_pair == "GBP_CAD":
					total_pnl += equity_curve[-1] * 50
				elif test_pair == "GBP_NZD":
					total_pnl += equity_curve[-1] * 1
				else:
					total_pnl += equity_curve[-1] * 100
			else:
				total_pnl += equity_curve[-1] 

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

