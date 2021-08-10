import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from sklearn.cluster import KMeans
from numpy import linalg as LA
from datetime import timedelta


import time
import datetime
import calendar
from dateutil import tz
import requests
import json
import copy

import math
import sys
import re
import xgboost as xgb

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import string
import random as rand
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from uuid import getnode as get_mac
import socket


import os
import mysql.connector

import logging
import os





def get_calendar_df(pair, year): 

	if pair != None:
		currencies = [pair[0:3], pair[4:7]]
	else:
		currencies = None

	if get_mac() == 150538578859218:
		with open("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt") as f:
			content = f.readlines()
	else:
		with open("/root/trading_data/calendar_" + str(year) + ".txt") as f:
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


def get_time_series(symbol, time, granularity="H1"):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=" + granularity + "&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	prices = []
	times = []

	index = 0
	while index < len(j):
		item = j[index]

		s = item['time']
		s = s[0 : s.index('.')]
		timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

		times.append(timestamp)
		prices.append(item['closeMid'])
		index += 1

	return prices, times


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

			if high != low:
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes

def calculate_time_diff(now_time, ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    date = datetime.datetime.utcfromtimestamp(now_time).strftime('%Y-%m-%d %H:%M:%S')
    e = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    def helper(d):
        if d.weekday() == 5:
            d += timedelta(days=1)
        return d.replace(hour=0, minute=0, second=0, microsecond=0)


    if e.weekday() in {5, 6}:
        e = helper(e)
    if s.weekday() in {5, 6}:
        s = helper(s)
    _diff = (e - s)
    while s < e:
        if s.weekday() in {5, 6}:
            _diff -= timedelta(days=1)
        elif s.weekday() == 0:
            s += timedelta(days=4)
        s += timedelta(days=1)

    return (_diff.total_seconds() / (60 * 60))

def cross_val_calculator(X, y, cross_val_num):

	y_true_indexes = [index for index in range(len(y)) if y[index] == True]
	y_false_indexes = [index for index in range(len(y)) if y[index] == False]


	y_test_all = []
	y_preds_all = []
	for iteration in range(cross_val_num):

		rand.shuffle(y_true_indexes)
		rand.shuffle(y_false_indexes)

		min_size = max(15, int(min(len(y_false_indexes), len(y_true_indexes)) * 0.35))
		if min_size >= max(len(y_true_indexes), len(y_false_indexes)) * 0.8:
			return -1

		true_indexes = y_true_indexes[:min_size]
		false_indexes = y_false_indexes[:min_size]

		X_train = []
		y_train = []

		X_test = []
		y_test = []
		for index in range(len(y)):
			if index in true_indexes + false_indexes:
				X_test.append(X[index])
				y_test.append(y[index])
			else:
				X_train.append(X[index])
				y_train.append(y[index])

		clf = xgb.XGBClassifier(seed=1)
		clf.fit(np.array(X_train), y_train)

		preds = clf.predict_proba(X_test)[:,1]

		y_test_all += y_test
		y_preds_all += list(preds)

	fpr, tpr, thresholds = metrics.roc_curve(y_test_all, y_preds_all)

	return metrics.auc(fpr, tpr)

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
for year in range(2007, 2020):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='mysql.newscaptial.com',
                              database='newscapital')

def back_test_news_calendar(curr_calendar_df, curr_time, select_pair, price_df):

	avg_barrier_auc = {}

	stat_dict = {}
	stat_dict["new_releases"] = []

	print (select_pair)
	
	
	for index, row in curr_calendar_df.iterrows():

		time_lag = calculate_time_diff(curr_time, row["time"])
		hour_lag = int(round(time_lag))
		if hour_lag > 24:
			continue

		if time_lag < 2:
			print ("< 1", time_lag, row["time"], curr_time)
			continue

		test_calendar = calendar_df[(abs(calendar_df["time"] - row["time"]) > 60 * 60 * 24 * 60) & (calendar_df["description"] == row["description"]) & (calendar_df["currency"] == row["currency"])]

		X_train = []
		y_train = []

		if select_pair[4:7] == "JPY":
			pip_size = 0.01
		else:
			pip_size = 0.0001

		pip_size *= 5

		start_time = time.time()

		y_train_map = {}
		X_train_map = {}

		for index2, calendar_row in test_calendar.iterrows():
			future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].head(hour_lag).values.tolist()
			if len(future_prices) < 4:
				continue

			feature1 = (calendar_row['actual'] - calendar_row['forecast'])
			feature2 = (calendar_row['actual'] - calendar_row['previous'])

			X_last = [feature1, feature2, future_prices[-1] - future_prices[0]]
			future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()

			time_decay_factor = 0.95
			for time_decay_index in range(1, 20):

				start_price = future_prices[hour_lag]
				hours_forecast = time_decay_index * 24

				last_price = future_prices[min(len(future_prices) - 1, hour_lag + hours_forecast)]

				if time_decay_index not in y_train_map:
					y_train_map[time_decay_index] = []
					X_train_map[time_decay_index] = []

				y_train_map[time_decay_index].append((last_price - start_price) > 0)
				X_train_map[time_decay_index].append(X_last)
					

		select_indexes = range(len(y_train))

		barrier_models = {}
		barrier_model_scores = {}
		barrier_model_biases = {}
		for time_decay_index in range(1, 20):

			if time_decay_index not in y_train_map:
				continue

			if len(y_train_map[time_decay_index]) < 50:
				continue

			cross_val_num = 10

			while cross_val_num > 4:
				try:
					score = cross_val_calculator(X_train_map[time_decay_index], y_train_map[time_decay_index], cross_val_num)
					break
				except:
					cross_val_num -= 1

			if cross_val_num <= 4:
				continue

			barrier_model_scores[time_decay_index] = score

			if time_decay_index not in avg_barrier_auc:
				avg_barrier_auc[time_decay_index] = []

			avg_barrier_auc[time_decay_index].append(score)

			barrier_clf = xgb.XGBClassifier(seed=1)
			barrier_clf.fit(np.array(X_train_map[time_decay_index]), y_train_map[time_decay_index])
			barrier_models[time_decay_index] = barrier_clf

		minte_lag = int(round(time_lag * 60))

		prices = price_df[(price_df["times"] <= curr_time) & (price_df["times"] >= row["time"])]["prices"].values.tolist()

		if len(prices) == 0:
			continue

		feature1 = (row['actual'] - row['forecast'])
		feature2 = (row['actual'] - row['previous'])

		X_last = [feature1, feature2, prices[-1] - prices[0]]

		end_time = time.time()

		barriers = []
		for time_decay_index in range(20):
			if time_decay_index not in barrier_models:
				continue

			prob = barrier_models[time_decay_index].predict_proba([X_last])[0][1]
			print ("Barrier", time_decay_index, prob, barrier_model_scores[time_decay_index], row["currency"])

			barriers.append({"barrier" : (5 + (5 * time_decay_index)), "prob" : prob, "auc" : barrier_model_scores[time_decay_index]})


			try:
				cursor = cnx.cursor()
				query = ("INSERT INTO historic_news_barrier_probs values ( \
					'" + row["currency"] + "', \
					'" + select_pair + "', \
					'" + str(curr_time) + "', \
					'" + str(row["time"]) + "', \
					'" + str(5 + (5 * time_decay_index)) + "', \
					'" + str(prob) + "', \
					'" + str(barrier_model_scores[time_decay_index]) + "', \
					'" + row["description"] + "', \
					'T0', \
					'" + str(prices[-1]) + "' \
					)")
				cursor.execute(query)
				cnx.commit()
				cursor.close()
			except:
				pass
		

		stat_dict["new_releases"].append({"barriers" : barriers, "time_lag" : time_lag, "currency" : calendar_row["currency"], "actual" : calendar_row['actual'], "forecast" : calendar_row['forecast'], "previous" : calendar_row['previous'], "pair" : select_pair, "description" : row["description"]})

import psutil

def checkIfProcessRunning(processName, command):
	count = 0
	#Iterate over the all the running process
	for proc in psutil.process_iter():
 
		try:
			cmdline = proc.cmdline()

			# Check if process name contains the given name string.
			if len(cmdline) > 3 and processName.lower() in cmdline[2] and command == cmdline[3]:
				count += 1
			elif len(cmdline) > 2 and processName.lower() in cmdline[1] and command == cmdline[2]:
				count += 1
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			pass

	if count >= 2:
		sys.exit(0)


select_pair = sys.argv[1]

checkIfProcessRunning('analyze_historic_news_signals_time_decay_binary.py', select_pair)

curr_time = calendar_df["time"].head(1).values.tolist()[0]
end_time = calendar_df["time"].tail(1).values.tolist()[0]

cursor = cnx.cursor()
query = ("SELECT max(time_stamp) FROM historic_news_barrier_probs where \
					currency_pair = '" + select_pair + "' and model_key='T0' \
					")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
	setup_rows.append(row1)

cursor.close()


if setup_rows[0][0] != None:
	curr_time = setup_rows[0][0]


#calendar_df = calendar_df[(select_pair[0:3] == calendar_df["currency"]) | (select_pair[4:7] == calendar_df["currency"])]

prices, times, volumes = load_time_series(select_pair, None, True)
buy_price_df = pd.DataFrame()
buy_price_df['times'] = times
buy_price_df["price_buy"] = prices
buy_price_df["volume_buy"] = volumes
buy_price_df.set_index('times', inplace=True)
buy_price_df.fillna(method='ffill', inplace=True)

prices, times, volumes = load_time_series(select_pair, None, False)
sell_price_df = pd.DataFrame()
sell_price_df['times'] = times
sell_price_df["price_sell"] = prices
sell_price_df["volume_sell"] = volumes
sell_price_df.set_index('times', inplace=True)
sell_price_df.fillna(method='ffill', inplace=True)

price_df = buy_price_df.join(sell_price_df)
price_df["prices"] = price_df.apply(lambda x: (x["price_buy"] + x["price_sell"]) * 0.5, axis=1)
price_df.reset_index(inplace=True)

while curr_time < end_time:

	date = datetime.datetime.utcfromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
	date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
	s = str(datetime.datetime.utcnow())
	s = s[0 : s.index('.')]
	week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

	if week_day_end == 6 or week_day_end == 0:
		start_time = curr_time - (60 * 60 * 24 * 3)
	else:
		start_time = curr_time - (60 * 60 * 24 * 1)

	curr_calendar_df = calendar_df[(calendar_df["time"] <= curr_time) & (calendar_df["time"] >= start_time)]

	if len(curr_calendar_df) > 0:
		back_test_news_calendar(curr_calendar_df, curr_time, select_pair, price_df)

	curr_time += 60 * 60 * 6

	cursor = cnx.cursor()
	query = ("SELECT time_stamp FROM historic_news_barrier_probs limit 1")
	cursor.execute(query)

	temp_rows = []
	for row1 in cursor:
		temp_rows.append(row1)

	cursor.close()



