


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
import re

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

def build_news_description_release_model(pair, year, df, description, price_df, equity, out_of_sample_year):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	test_calendar = df[df['description'] == description]

	X_train = []
	y_train = []

	X_test = []
	y_test = []
	for index, row in test_calendar.iterrows():

		if row["year"] == out_of_sample_year:
			continue

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
			equity += abs(y2) - commission
		else:
			equity -= abs(y2) + commission

		equity_curve.append(equity)
		

	return equity_curve, equity, clf_mean

def out_of_sample_back_test(model_whilelist, year, df, price_df):

	equity = 0
	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	model_map = {}
	for item in model_whilelist:
		model_map[item["description"]] = item["model"]

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	test_calendar = df[df['year'] == year]

	X_train = []
	y_train = []

	X_test = []
	y_test = []
	for index, row in test_calendar.iterrows():

		if row["description"] not in model_map:
			continue

		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(48).values.tolist()[:12]
		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(48).values.tolist()[:12]

		if len(future_prices) < 12:
			continue

		prediction = model_map[row["description"]].predict([[row['actual'] - row['forecast'], row['actual'] - row['previous']]])[0]

		y2 = future_prices[-1] - future_prices[0]

		if (prediction > 0) == (y2 > 0):
			equity += abs(y2) - commission
		else:
			equity -= abs(y2) + commission

	return equity




out_of_sample_equity = 0
for pair in currency_pairs:

	out_of_sample_year = 2016

	df_set = []
	for year in range(2007, 2018):

		df = get_calendar_df(pair, year)
		df["year"] = year

		df_set.append(df)

	df = pd.concat(df_set)

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

	sharpes = []
	model_whilelist = []
	for description in description_count_map:
		if description_count_map[description] > 60:
			print pair, description, description_count_map[description]

			equity = 0
			equity_curve = []
			for year in range(2007, 2018):
				if year == out_of_sample_year:
					continue

				try:
					ec, equity, model = build_news_description_release_model(pair, year, df, description, price_df, equity, out_of_sample_year)
					equity_curve += ec
				except:
					pass

			returns = []
			for index in range(len(equity_curve) - 1):
				returns.append(equity_curve[index+1] - equity_curve[index])

			sharpes.append(np.mean(returns) / np.std(returns))

			
			if sharpes[-1] > 0.15:
				ec, equity, model = build_news_description_release_model(pair, None, df, description, price_df, equity, out_of_sample_year)
				model_whilelist.append({"model" : model, "sharpe" : sharpes[-1], "description" : description, "pair" : pair})
			

			print sharpes[-1]

			print "Mean Sharpe", np.mean(sharpes)

	pickle.dump(model_whilelist, open("/tmp/new_release_trend_" + pair, 'wb'))


	final_equity = out_of_sample_back_test(model_whilelist, out_of_sample_year, df, price_df)
	if pair[4:7] != "JPY":
		final_equity *= 100

	out_of_sample_equity += final_equity
	print "final equity", pair, final_equity, out_of_sample_equity


