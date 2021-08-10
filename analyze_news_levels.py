


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


def skipForward(time):
	    day = datetime.fromtimestamp(time).strftime("%A")

	    if day == 'Saturday':
	        return time + (60 * 60 * 24 * 2)

	    if day == 'Sunday':
	        return time + (60 * 60 * 24)

	    return time


def findCalendarFeatures(self, pair, start, end, calendar, training_hours):

    snapshot = calendar[(calendar.time >= start) & (calendar.time < end)]

    first_currency = pair[0:3]
    second_currency = pair[4:7]

    currencies = [first_currency, second_currency]

    feature_vector = [0.0] * 24

    count = 0
    for currency in currencies:
    	count += 1
        usd = snapshot[snapshot.currency == currency]
        if len(usd) == 0:
            continue

        for index, row in usd.iterrows():

            hour = int(float(row['time'] - start) / (60 * 60))

            impact = int(row['impact'])
            if impact == 3:
            	weight = 1.0
            else:
            	weight = 0.0

            feature_vector[hour] += impact

    return feature_vector

def get_calendar_df(pair, year): 

	currencies = [pair[0:3], pair[4:7]]

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

		if toks[2] in currencies:

			est = datetime.datetime.strptime(toks[0] + " " + toks[1], "%b%d.%Y %H:%M%p")
			est = est.replace(tzinfo=from_zone)
			utc = est.astimezone(to_zone)

			time = calendar.timegm(utc.timetuple())

			contents.append([toks[2], time])

	return pd.DataFrame(contents, columns=["currency", "time"])


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

currency_pairs = ["AUD_CHF"
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


		for future_time, time_index in zip(future_times, range(len(future_times))):

			prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()
			times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()

			for price_v, time_v in zip(prices, times):

				if time_v <= future_time:
					price_deltas.append((price_v - first_price) / 1)

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


def back_test(currency1, currency2, price_df, pair, year, test_calendar):


	equity = 0
	equity_curve = []
	orders = []

	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		before_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24 * 20).values.tolist()
		before_times = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 20).values.tolist()

		calendar_times = test_calendar[(test_calendar['time'] >= before_prices[0]) & (test_calendar['time'] <= row['time'])]['time'].values.tolist()

		kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1, 
		           random_state = 42).fit([[v] for v in before_prices])

		price_level_map = {}
		for calendar_time in calendar_times:

			after_prices = (price_df[price_df['times'] >= calendar_time])['prices'].head(12).values.tolist()
			if len(after_prices) == 0:
				continue

			cluster_id = kmeans.predict([[after_prices[0]]])[0]

			if cluster_id not in price_level_map:
				price_level_map[cluster_id] = []

			for after_price in after_prices[1:]:
				delta = after_price - after_prices[0]
				price_level_map[cluster_id].append(delta)

		cluster_id = kmeans.predict([[after_prices[0]]])[0]
		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()
		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()
		first_price = future_prices[0]
		future_times = future_times[12:]
		future_prices = future_prices[12:]

		if cluster_id not in price_level_map:
			continue

		deltas = price_level_map[cluster_id]

		for future_price, future_time in zip(future_prices, future_times):

			delta = ((future_price - first_price) - np.mean(deltas)) / np.std(deltas)

			if abs(delta) > 2:
				new_order = Order()
				new_order.open_price = future_price
				new_order.open_time = future_time
				new_order.dir = delta > 0
				new_order.amount = abs(delta)#abs(delta)
				orders.append(new_order)

			total_pnl = 0
			order_count = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					total_pnl += abs(future_price - order.open_price) * order.amount
				else:
					total_pnl -= abs(future_price - order.open_price) * order.amount

				order_count += 1

			if total_pnl > 0 or order_count >= 8:
				equity += total_pnl
				total_pnl = 0
				orders = []

		equity += total_pnl
		orders = []
		print equity
			
		equity_curve.append(equity)

	return equity_curve


total_pnl = 0
for currency_pair in currency_pairs:
	print currency_pair

	year = 2017


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

		test_calendar = get_calendar_df(test_pair, year)

		prices, times = load_time_series(test_pair, None)

		price_df = pd.DataFrame()
		price_df['prices'] = prices
		price_df['times'] = times

				
		equity_curve = back_test(first_currency, second_currency, price_df, test_pair, year, test_calendar)

		if test_pair[4:7] != "JPY":
			total_pnl += equity_curve[-1] * 100
		else:
			total_pnl += equity_curve[-1]

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

