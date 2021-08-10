


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

from sklearn.cluster import SpectralClustering
from sklearn.ensemble import GradientBoostingRegressor
import download_calendar as download_calendar
import re


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


def cluster_news_release_fvs(description, df, year):

	calendar = df[df["description"] == description]

	X_train = []
	for index, row in calendar.iterrows():

		if year == row["year"]:
			continue

		feature_vector = [row['actual'] - row['forecast'], row['actual'] - row['previous']]

		X_train.append(feature_vector)

	return KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, 
		           random_state = 42).fit(X_train)

def cluster_price_trends(price_df, pair, year, test_calendar):

	test_calendar = get_calendar_df(pair, year)

	price_deltas = []
	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()[:24]

		if len((price_df[price_df['times'] >= row['time']])) == 0:
			continue

		price_deltas.append([future_prices[-1] - future_prices[0]])


	return KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, 
		           random_state = 42).fit(price_deltas)


def group_news_event_by_price_trend(df, year, pair, price_df, price_delta_kmeans, news_event_kmeans_map, description_map_offset):

	prev_time = None
	price_delta_map = {}
	for index, row in df.iterrows():

		if row['time'] == prev_time:
			continue

		if row["description"] not in description_map_offset:
			continue

		prev_time = row['time']

		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()[:24]

		if len((price_df[price_df['times'] >= row['time']])) == 0:
			continue

		price_delta_cluster_id = price_delta_kmeans.predict([[future_prices[-1] - future_prices[0]]])[0]

		description_offset = description_map_offset[row["description"]]
		news_event_cluster_id = news_event_kmeans_map[row["description"]].predict([[row['actual'] - row['forecast'], row['actual'] - row['previous']]])[0]
		news_event_cluster_id += description_offset

		if price_delta_cluster_id not in price_delta_map:
			price_delta_map[price_delta_cluster_id] = []

		price_delta_map[price_delta_cluster_id].append(news_event_cluster_id)

	return price_delta_map

def process_news_similarity(df, year, price_delta_map, total_nodes):

	association_map_count = {}
	association_map_nodes = {}
	for key in price_delta_map:
		news_event_ids = price_delta_map[key]

		for news_i in news_event_ids:
			for news_j in news_event_ids:

				if news_i == news_j:
					continue

				if news_i >= total_nodes or news_j >= total_nodes:
					print news_i, news_j, total_nodes

				map_key = str(news_i) + " "  + str(news_j)
				if map_key not in association_map_count:
					association_map_count[map_key] = 0
					association_map_nodes[map_key] = [news_i, news_j]

				association_map_count[map_key] += 1

	association_matrix = [[0] * total_nodes] * total_nodes
	for key in association_map_nodes:

		nodes = association_map_nodes[key]
		association_matrix[nodes[0]][nodes[1]] = association_map_count[key]

	cluster_offset = total_nodes
	cluster_num = int(total_nodes / 2)
	node_id_map = {}

	while cluster_num > 5:

		print "spectral cluster", cluster_num
		clustering = SpectralClustering(n_clusters=cluster_num,
		         assign_labels="discretize",
		         random_state=0).fit(association_matrix)

		
		for label, index in zip(clustering.labels_, range(len(clustering.labels_))):

			if index not in node_id_map:
				node_id_map[index] = [index]

			node_id_map[index].append(label + cluster_offset)

		cluster_offset += cluster_num
		cluster_num /= 2

	return node_id_map, cluster_offset

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

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

def build_news_description_release_model(year, df, description, price_df, equity):

	test_calendar = df[df['description'] == description]

	X_train = []
	y_train = []

	X_test = []
	y_test = []
	for index, row in test_calendar.iterrows():

		#print price_df[price_df['times'] >= row['time']]

		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()[:12]
		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()[:12]

		if len(future_prices) < 12:
			continue

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]

		if row['year'] != year:
			X_train.append([row['actual'] - row['forecast'], row['actual'] - row['previous']])
			y_train.append(future_prices[-1] - future_prices[0])
		else:
			X_test.append([row['actual'] - row['forecast'], row['actual'] - row['previous']])
			y_test.append(future_prices[-1] - future_prices[0])

	if len(X_train) < 40:
		return [], equity, None

	clf_mean = GradientBoostingRegressor(random_state=42)
	clf_mean.fit(X_train, y_train)

	if year == None:
		return [], equity, clf_mean

	equity_curve = []
	predictions = clf_mean.predict(X_test)
	for y1, y2 in zip(predictions, y_test):

		if (y1 > 0) == (y2 > 0):
			equity += abs(y2)
		else:
			equity -= abs(y2)

		equity_curve.append(equity)
		

	return equity_curve, equity, clf_mean

def back_test_year(year, df, pair, returns):

	description_count_map = {}
	descriptions = df["description"].values.tolist()

	prices, times = load_time_series(pair, None)
	price_df = pd.DataFrame()
	price_df['prices'] = prices
	price_df['times'] = times

	for description in descriptions:

		if description not in description_count_map:
			description_count_map[description] = 0

		description_count_map[description] += 1

	if os.path.isfile("/tmp/description_kmeans_" + str(year) + "_" + pair) == False:
		description_kmeans = {}
		for description in description_count_map:
			if description_count_map[description] > 30:

				print pair, description, description_count_map[description]
				description_kmeans[description] = cluster_news_release_fvs(description, df, year)

		pickle.dump(description_kmeans, open("/tmp/description_kmeans_" + str(year) + "_" + pair, 'wb'))
	else:
		description_kmeans = pickle.load(open("/tmp/description_kmeans_" + str(year) + "_" + pair, 'rb'))


	description_offset = 0
	description_map_offset = {}
	for description in description_kmeans:
		description_map_offset[description] = description_offset
		description_offset += len(description_kmeans[description].cluster_centers_)

	if os.path.isfile("/tmp/node_id_map" + str(year) + "_" + pair) == False:
		price_delta_kmeans = cluster_price_trends(price_df, pair, year, df)

		price_delta_map = group_news_event_by_price_trend(df, year, pair, price_df, price_delta_kmeans, description_kmeans, description_map_offset)

		node_id_map, cluster_offset = process_news_similarity(df, year, price_delta_map, description_offset)

		pickle.dump(node_id_map, open("/tmp/node_id_map" + str(year) + "_" + pair, 'wb'))
		pickle.dump(cluster_offset, open("/tmp/cluster_offset" + str(year) + "_" + pair, 'wb'))
	else:
		node_id_map = pickle.load(open("/tmp/node_id_map" + str(year) + "_" + pair, 'rb'))
		cluster_offset = pickle.load(open("/tmp/cluster_offset" + str(year) + "_" + pair, 'rb'))


	if os.path.isfile("/tmp/clf" + str(year) + "_" + pair) == False or True:
		calendar = df[df["year"] != year]

		X_train = []
		y_train = []
		for index, row1 in calendar.iterrows():
			if row1["description"] not in description_kmeans:
				continue

			future_prices = (price_df[price_df['times'] >= row1['time']])['prices'].head(48).values.tolist()[:24]
			seg_calendar = calendar[calendar["time"] <= row1["time"]].tail(1)
			train_vector = [0] * cluster_offset

			for index, row in seg_calendar.iterrows():
				if row["description"] not in description_kmeans:
					continue

				feature_vector = [row['actual'] - row['forecast'], row['actual'] - row['previous']]
				

				kmeans = description_kmeans[row["description"]]

				cluster_id = kmeans.predict([feature_vector])[0]
				cluster_id += description_map_offset[row["description"]]

				proxy_ids = node_id_map[cluster_id]

				
				for proxy_id in proxy_ids:
					train_vector[proxy_id] = 1.0
				
			X_train.append(train_vector)
			y_train.append(future_prices[-1] - future_prices[0])

		clf = GradientBoostingRegressor(random_state=42)
		clf.fit(X_train, y_train)

		pickle.dump(clf, open("/tmp/clf" + str(year) + "_" + pair, 'wb'))
	else:
		clf = pickle.load(open("/tmp/clf" + str(year) + "_" + pair, 'rb'))

	calendar = df[df["year"] == year]


	equity = 0
	orders = []
	returns = []
	for index, row1 in calendar.iterrows():
		if row1["description"] not in description_kmeans:
			continue

		future_prices = (price_df[price_df['times'] >= row1['time']])['prices'].head(48).values.tolist()[:24]
		seg_calendar = calendar[calendar["time"] <= row1["time"]].tail(1)
		train_vector = [0] * cluster_offset

		for index, row in seg_calendar.iterrows():
			if row["description"] not in description_kmeans:
				continue

			feature_vector = [row['actual'] - row['forecast'], row['actual'] - row['previous']]
			

			kmeans = description_kmeans[row["description"]]

			cluster_id = kmeans.predict([feature_vector])[0]
			cluster_id += description_map_offset[row["description"]]

			proxy_ids = node_id_map[cluster_id]

			
			for proxy_id in proxy_ids:
				train_vector[proxy_id] = 1.0
			
		prediction = clf.predict([train_vector])[0]

		total_pnl = 0
		new_orders = []
		
		for order in orders:

			if order.dir != (prediction > 0):

				if (order.dir) == (future_prices[0] > order.open_price):
					equity += abs(future_prices[0] - order.open_price)
					returns.append(abs(future_prices[0] - order.open_price))
				else:
					equity -= abs(future_prices[0] - order.open_price)
					returns.append(-abs(future_prices[0] - order.open_price))

				continue

			if (order.dir) == (future_prices[0] > order.open_price):
				total_pnl += abs(future_prices[0] - order.open_price)
			else:
				total_pnl -= abs(future_prices[0] - order.open_price)

			new_orders.append(order)

		orders = new_orders

		if abs(prediction) > 0.001:
			order = Order()
			order.open_price = future_prices[0]
			order.dir = (prediction > 0)
			order.amount = 1
			orders.append(order)

		print "Equity", equity + total_pnl

	
	return returns


for pair in ["EUR_USD"]:

	df_set = []
	for year in range(2007, 2018):

		df = get_calendar_df(pair, year)
		df["year"] = year

		df_set.append(df)

	df = pd.concat(df_set)

	returns = []
	for year in range(2007, 2011):
		returns += back_test_year(year, df, pair, returns)

		print "sharpe", np.mean(returns) / np.std(returns)


