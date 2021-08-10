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
from sklearn.model_selection import cross_val_score
import gzip, cPickle
import string
import random as rand

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

import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import logging
from close_trade import *
import os


import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_calendar_df(pair, year): 

	if pair != None:
		currencies = [pair[0:3], pair[4:7]]
	else:
		currencies = None

	with open("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt") as f:
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
	onlyfiles = [f for f in listdir('/Users/andrewstevens/Downloads/economic_calendar/') if isfile(join('/Users/andrewstevens/Downloads/economic_calendar/', f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Candlestick_1_Hour_BID' in file:
			break

	with open('/Users/andrewstevens/Downloads/economic_calendar/' + file) as f:
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

def find_z_score(currency_pair, test_calendar, time_lag, price_arc, features, model_map, year, price_df):

	if time_lag not in model_map:
		X_train = []
		y_train = []
		for index2, calendar_row in test_calendar.iterrows():

			if calendar_row["year"] == year:
				continue

			future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].head(time_lag).values.tolist()
			if len(future_prices) < 12:
				continue

			feature1 = (calendar_row['actual'] - calendar_row['forecast'])
			feature2 = (calendar_row['actual'] - calendar_row['previous'])

			if calendar_row["currency"] == currency_pair[0:3]:
				X_train.append([feature1, feature2, 0, 0])
			else:
				X_train.append([0, 0, feature1, feature2])

			y_train.append(future_prices[-1] - future_prices[0])

		select_indexes = range(len(y_train))

		model_set = []
		for model_index in range(30):

			rand.shuffle(select_indexes)
			chosen_indexes = select_indexes[:int(len(select_indexes) * 0.8)]

			y = [y_train[i] for i in chosen_indexes]
			x = [X_train[i] for i in chosen_indexes]

			clf_mean = GradientBoostingRegressor(random_state=42)
			clf_mean.fit(x, y)

			model_set.append(clf_mean)

		model_map[time_lag] = model_set

	model_set = model_map[time_lag]

	returns = []
	for model in model_set:
		forecast = model.predict([features])[0]

		returns.append(forecast)

	std = np.std(returns)
	mean = np.mean(returns)

	z_score = (price_arc - mean) / std
	return z_score, model_map

def back_test_news_model(test_calendar, price_df, year, pair, description_map):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	y_train = []
	X_train = []

	X_test = []
	y_test = []
	for index2, calendar_row in test_calendar.iterrows():

		last_time = (price_df[price_df['times'] <= calendar_row['time']])['times'].tail(5 * 24).values.tolist()[0]
		weekly_calendar = (test_calendar[(test_calendar['time'] >= last_time) & (test_calendar['time'] <= calendar_row["time"])])

		future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()
		future_times = (price_df[price_df['times'] >= calendar_row['time']])['times'].values.tolist()

		time_index = 0
		start_price = future_prices[0]
		label_vector = []
		for barrier_index in range(1, 30):

			top_barrier = start_price + (pip_size + (pip_size * barrier_index))
			bottom_barrier = start_price - (pip_size + (pip_size * barrier_index))

			found = False
			for price in future_prices[time_index:]:
				
				if price >= top_barrier:
					label_vector.append(1)
					found = True
					break

				if price <= bottom_barrier:
					label_vector.append(-1)
					found = True
					break

			if found == False:
				label_vector.append(0)

		if calendar_row["year"] == year:
			y_test.append(label_vector)
		else:
			y_train.append(label_vector)

		feature_vector = [0] * ((len(description_map) * 4))
		for index3, train_row in weekly_calendar.iterrows():
			feature1 = (train_row['actual'] - train_row['forecast'])
			feature2 = (train_row['actual'] - train_row['previous'])

			offset = description_map[train_row["description"]] * 4
			#print description_map[train_row["description"]], len(description_map) * 4, len(feature_vector), train_row["description"]

			if train_row["currency"] == pair[0:3]:
				feature_vector[offset] = feature1
				feature_vector[offset+1] = feature2
			else:
				feature_vector[offset+2] = feature1
				feature_vector[offset+3] = feature2

		if calendar_row["year"] == year:
			X_test.append(feature_vector)
		else:
			X_train.append(feature_vector)

	model = Sequential()
	model.add(Dense(len(feature_vector), input_dim = len(feature_vector) , activation = 'relu'))
	model.add(Dense(len(feature_vector), activation = 'relu'))
	model.add(Dense(len(feature_vector), activation = 'relu'))
	model.add(Dense(len(feature_vector), activation = 'relu'))
	model.add(Dense(len(y_train[0]), activation = 'softmax'))

	model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

	model.fit(np.array(X_train), np.array(y_train), epochs = 10, batch_size = 2)

	pickle.dump(model, open("/tmp/news_nn_" + pair + ".pickle", "wb"))

	return model

	predictions = model.predict(np.array(X_test))

	dots = []
	for y, p in zip(y_test, predictions):
		dot_prod = np.dot(y, p)
		norm1 = np.linalg.norm(y)
		norm2 = np.linalg.norm(p)

		dots.append(dot_prod / (norm1 * norm2))

	print pair, "mean", np.mean(dots)


	#return clf, X_test, y_test, (0.001 + (0.001 * best_score_index)), best_score


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


for currency_pair in currency_pairs:#, "AUD_CAD", "NZD_CAD", "AUD_NZD", "EUR_GBP", "EUR_USD", "USD_CAD", "GBP_USD"]:
	description_map = {}
	for year1 in range(2007, 2019):
		descriptions = get_calendar_df(currency_pair, year1)["description"].values.tolist()
		for description in descriptions:
				if description not in description_map:
					description_map[description] = len(description_map)


	if currency_pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[currency_pair] * pip_size

	test_calendar = calendar_df[(calendar_df["currency"] == currency_pair[0:3]) | (calendar_df["currency"] == currency_pair[4:7])]

	if len(test_calendar) < 25:
		continue

	print currency_pair, description

	prices, times = load_time_series(currency_pair, None)

	price_df = pd.DataFrame()
	price_df['times'] = times
	price_df["prices"] = prices
	price_df['mult_factor'] = 1.0
	price_df.set_index('times', inplace=True)

	price_df.fillna(method='ffill', inplace=True)
	price_df.reset_index(inplace=True)

	total_profit = 0
	for year in [None]:#range(2007, 2019):

		back_test_news_model(test_calendar, price_df, year, currency_pair, description_map)




