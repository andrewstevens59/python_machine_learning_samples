import matplotlib.pyplot as plt
import networkx as nx
import pickle


import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.model_selection import cross_val_score


def draw_class_graph(edge_map):
	G = nx.Graph()

	total_usage_count = 0
	total_no_usage_count = 0
	for edge in edge_map.values():
		total_usage_count += edge["usage_count"] 
		total_no_usage_count += edge["no_usage_count"] 

	total_diff = 0
	for edge in edge_map.values():
		edge["usage_count"] = float(edge["usage_count"]) / total_usage_count
		edge["no_usage_count"] = float(edge["no_usage_count"]) / total_no_usage_count

		total_diff += abs(edge["usage_count"] - edge["no_usage_count"])

	for edge in edge_map.values():

		if edge["usage_count"] > edge["no_usage_count"]:
			G.add_edge(edge["src"], edge["dst"], color='lime', weight=100 * (float(edge["usage_count"] - edge["no_usage_count"]) / total_diff))
		else:
			G.add_edge(edge["src"], edge["dst"], color='orange', weight=100 * (float(edge["no_usage_count"] - edge["usage_count"]) / total_diff))

	pos = nx.random_layout(G)

	edges = G.edges()
	colors = [G[u][v]['color'] for u,v in edges]
	weights = [G[u][v]['weight'] for u,v in edges]

	pos = nx.spring_layout(G, iterations=10)
	nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels = True)
	plt.show()

def prob_density_graph(edge_map):

	G = nx.Graph()

	total_count = 0
	for edge in edge_map.values():
		total_count += edge["usage_count"] 
		total_count += edge["no_usage_count"] 

	for edge in edge_map.values():

		edge_weight = edge["usage_count"] + edge["no_usage_count"]


		G.add_edge(edge["src"], edge["dst"], color='black', weight=100 * (float(edge_weight) / total_count))

	pos = nx.random_layout(G)

	edges = G.edges()
	colors = [G[u][v]['color'] for u,v in edges]
	weights = [G[u][v]['weight'] for u,v in edges]

	pos = nx.spring_layout(G, iterations=10)
	nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels = True)
	plt.title("Transition Probability Density")
	plt.show()

def error_graph(edge_map):

	G = nx.Graph()

	total_usage_count = 0
	total_no_usage_count = 0
	for edge in edge_map.values():
		total_usage_count += edge["usage_count"] 
		total_no_usage_count += edge["no_usage_count"] 

	total_diff = 0
	for edge in edge_map.values():
		edge["usage_count"] = float(edge["usage_count"]) / total_usage_count
		edge["no_usage_count"] = float(edge["no_usage_count"]) / total_no_usage_count
		total_diff += abs(edge["usage_count"] - edge["no_usage_count"])

	total_count = 0
	for edge in edge_map.values():
		total_count += edge["usage_count"] 
		total_count += edge["no_usage_count"] 

	total_edge_diff = 0
	for edge in edge_map.values():

		edge_weight1 = (float(edge["usage_count"] + edge["no_usage_count"]) / total_count)
		edge_weight2 = (float(edge["usage_count"] - edge["no_usage_count"]) / total_diff)
		total_edge_diff += abs(edge_weight2 - edge_weight1)

	for edge in edge_map.values():
		edge_weight1 = (float(edge["usage_count"] + edge["no_usage_count"]) / total_count)
		edge_weight2 = (float(edge["usage_count"] - edge["no_usage_count"]) / total_diff)

		print (float(abs(edge_weight2 - edge_weight1)) / total_edge_diff)
		G.add_edge(edge["src"], edge["dst"], color='red', weight=100 * (float(abs(edge_weight2 - edge_weight1)) / total_edge_diff))

	pos = nx.random_layout(G)

	edges = G.edges()
	colors = [G[u][v]['color'] for u,v in edges]
	weights = [G[u][v]['weight'] for u,v in edges]

	pos = nx.spring_layout(G, iterations=10)
	nx.draw(G, pos, edges=edges, edge_color=colors, width=weights, with_labels = True)
	plt.title("Transition Probability Density")
	plt.show()


import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
import download_calendar as download_calendar
import gzip, cPickle
import string
import random as rand

import os
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LinearRegression
from maximize_sharpe import *

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from uuid import getnode as get_mac
import socket
import paramiko
import json


import os



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

				contents.append([toks[2], time, toks[3], toks[4], toks[5], toks[6], year])
			except:
				pass

	return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "year"])


def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=America")

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

def back_test_news_model(test_calendar, price_df, year, pair):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	y_train_map = {}
	X_train = []

	X_test = []
	y_test_map = {}
	for index2, calendar_row in test_calendar.iterrows():

		future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()
		future_times = (price_df[price_df['times'] >= calendar_row['time']])['times'].values.tolist()

		back_prices = (price_df[price_df['times'] <= calendar_row['time']])['prices'].values.tolist()[:12]
		back_times = (price_df[price_df['times'] <= calendar_row['time']])['times'].values.tolist()[:12]

		if len(future_prices) < 12 or len(back_prices) < 12:
			continue

		deltas = []
		for back_index in range(len(back_times)):
			deltas.append(back_prices[back_index] - back_prices[0])

		mean = np.mean(deltas)
		std = np.std(deltas)

		z_score = ((future_prices[11] - future_prices[0]) - mean) / std

		for time_index in range(12, 24):

			start_price = future_prices[time_index]
			start_time = future_times[time_index]

			for barrier_index in range(1, 15):

				if barrier_index not in y_train_map:
					y_train_map[barrier_index] = []
					y_test_map[barrier_index] = []

				top_barrier = start_price + (0.001 + (0.001 * barrier_index))
				bottom_barrier = start_price - (0.001 + (0.001 * barrier_index))

				found = False
				for price in future_prices[time_index:]:
					
					if price >= top_barrier:

						if calendar_row["year"] == year:
							y_test_map[barrier_index].append(start_time)
						else:
							y_train_map[barrier_index].append(True)
						found = True
						break

					if price <= bottom_barrier:
						if calendar_row["year"] == year:
							y_test_map[barrier_index].append(start_time)
						else:
							y_train_map[barrier_index].append(False)
						found = True
						break

			if found:
				if calendar_row["year"] == year:
					X_test.append(z_score)
				else:
					X_train.append(z_score)

			else:
				print "no"

	return X_train, y_train_map

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


df_set = []
for year in range(2007, 2019):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

def create_graph(pair):
	X_train = pickle.load(open("/tmp/X_train" + pair, 'rb'))
	y_train = pickle.load(open("/tmp/y_train" + pair, 'rb'))

	X_sample = [[X_train[index]] for index in range(len(X_train))]

	kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1, 
	               random_state = 42).fit(X_sample) 

	predictions = kmeans.predict(X_sample)

	edge_map = {}
	vertex_map = {}
	total_usage_count = 0
	total_no_usage_count = 0

	total_vertex_usage_count = 0
	total_no_vertex_usage_count = 0
	for prediction_index in range(len(predictions) - 1):

		src = predictions[prediction_index]
		dst = predictions[prediction_index + 1]

		label = y_train[1][prediction_index + 1]

		if src not in vertex_map:
			vertex_map[src] = {}
			vertex_map[src]["usage_count"] = 0
			vertex_map[src]["no_usage_count"] = 0

		if y_train[1][prediction_index]:
			vertex_map[src]["usage_count"] += 1
			total_vertex_usage_count += 1
		else:
			vertex_map[src]["no_usage_count"] += 1
			total_no_vertex_usage_count += 1

		key = str(src) + "_" + str(dst)

		if key not in edge_map:
			edge_map[key] = {}
			edge_map[key]["src"] = src
			edge_map[key]["dst"] = dst

			edge_map[key]["usage_count"] = 0
			edge_map[key]["no_usage_count"] = 0

		if label:
			edge_map[key]["usage_count"] += 1
			total_usage_count += 1
		else:
			edge_map[key]["no_usage_count"] += 1
			total_no_usage_count += 1

	separation_score = 0
	for edge in edge_map.values():
		score = abs((float(edge["usage_count"]) / total_usage_count)  - (float(edge["no_usage_count"]) / total_no_usage_count))
		separation_score += score
		print score

	print "Edge Separation Score", separation_score

	separation_score = 0
	for vertex in vertex_map.values():
		separation_score += abs((float(vertex["usage_count"]) / total_vertex_usage_count)  - (float(vertex["no_usage_count"]) / total_no_vertex_usage_count))

	print "Vertex Separation Score", separation_score

	draw_class_graph(edge_map)

from sklearn.linear_model import LogisticRegression

def build_model(pair):
	X_train = pickle.load(open("/tmp/X_train" + pair, 'rb'))
	y_train = pickle.load(open("/tmp/y_train" + pair, 'rb'))

	X_sample = [[X_train[index]] for index in range(len(X_train))]

	kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1, 
	               random_state = 42).fit(X_sample) 

	predictions = kmeans.predict(X_sample)

	edge_map = {}
	vertex_map = {}
	total_usage_count = 0
	total_no_usage_count = 0

	total_vertex_usage_count = 0
	total_no_vertex_usage_count = 0

	edge_map = {}
	edge_history = []

	for prediction_index in range(len(predictions) - 1):

		src = predictions[prediction_index]
		dst = predictions[prediction_index + 1]

		key = str(src) + "_" + str(dst)
		if key not in edge_map:
			edge_map[key] = len(edge_map)

	X = []
	y = []
	for prediction_index in range(len(predictions) - 1):

		src = predictions[prediction_index]
		dst = predictions[prediction_index + 1]

		label = y_train[10][prediction_index + 1]

		key = str(src) + "_" + str(dst)
		if key not in edge_map:
			edge_map[key] = len(edge_map)

		edge_history.append(edge_map[key])
		if len(edge_history) >= 1:

			feature_vector = [0] * len(edge_map)
			for edge in edge_history:
				feature_vector[edge] += 1

			X.append(feature_vector)
			y.append(label)
			edge_history = edge_history[1:]

	clf = GradientBoostingClassifier()
	scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
	score = np.mean(scores)
	print "Mean AUC", score

build_model("AUD_CAD")

sys.exit(0)


for currency_pair in ["EUR_USD"]:

	description_map = {}
	for year1 in range(2007, 2019):
		descriptions = get_calendar_df(currency_pair, year1)["description"].values.tolist()
		for description in descriptions:
				description_map[description] = len(description_map)


	if currency_pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[currency_pair] * pip_size

	test_calendar = calendar_df[(calendar_df["currency"] == currency_pair[0:3]) | (calendar_df["currency"] == currency_pair[4:7])]

	if len(test_calendar) < 25:
		continue

	prices, times = load_time_series(currency_pair, None)

	price_df = pd.DataFrame()
	price_df['times'] = times
	price_df["prices"] = prices
	price_df['mult_factor'] = 1.0
	price_df.set_index('times', inplace=True)

	price_df.fillna(method='ffill', inplace=True)
	price_df.reset_index(inplace=True)

	X_train, y_train = back_test_news_model(test_calendar, price_df, year, currency_pair)


	pickle.dump(X_train, open("/tmp/X_train" + currency_pair, 'wb'))
	pickle.dump(y_train, open("/tmp/y_train" + currency_pair, 'wb'))
	sys.exit(0)





