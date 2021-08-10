import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from sklearn.cluster import KMeans
from numpy import linalg as LA

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

	with open(prefix + file) as f:
	    content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	from_zone = tz.tzlocal()
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
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes

def calculate_time_diff(ts, curr_time):

	date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

	week_day_start = date.weekday()

	time_start = calendar.timegm(date.timetuple())

	s = str(datetime.datetime.utcnow())
	s = s[0 : s.index('.')]

	week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

	time_end = curr_time

	time_diff_hours = time_end - time_start

	time_diff_hours /= 60 * 60

	week_num = int(time_diff_hours / (24 * 7))

	
	if time_diff_hours >= 48:
		if week_num >= 1:
			time_diff_hours -= 48 * week_num

		if week_day_start != 6 and week_day_end < week_day_start:
			time_diff_hours -= 48
		elif week_day_end == 6 and week_day_end > week_day_start:
			time_diff_hours -= 48
	

	return time_diff_hours

def cross_val_calculator(instance_offsets, X, y, cross_val_num, time_lag):

	y_true_indexes = [index for index in range(len(instance_offsets)-1) if y[min(instance_offsets[index+1] - 1, instance_offsets[index] + time_lag)] == True]
	y_false_indexes = [index for index in range(len(instance_offsets)-1) if y[min(instance_offsets[index+1] - 1, instance_offsets[index] + time_lag)] == False]

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
		for index in range(len(instance_offsets)-1):
			if index in true_indexes + false_indexes:
				X_test.append(X[min(instance_offsets[index+1] - 1, instance_offsets[index] + time_lag)])
				y_test.append(y[min(instance_offsets[index+1] - 1, instance_offsets[index] + time_lag)])
			else:
				X_train += [X[min(instance_offsets[index+1] - 1, instance_offsets[index] + time_lag)]]
				y_train += [y[min(instance_offsets[index+1] - 1, instance_offsets[index] + time_lag)]]

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
for year in range(2007, 2019):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='66.33.203.187',
                              database='newscapital')

auc_scores = []

def back_test_news_calendar(curr_calendar_df, curr_time, select_pair, price_df):

	avg_barrier_auc = {}

	stat_dict = {}
	stat_dict["new_releases"] = []

	print (select_pair)
	
	
	for index, row in curr_calendar_df.iterrows():

		time_lag = calculate_time_diff(row["time"], curr_time)
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
		instance_offset = {}

		for index2, calendar_row in test_calendar.iterrows():
			future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()
			if len(future_prices) < 4:
				continue

			for time_offset in range(1, min(24, len(future_prices))):

				feature1 = (calendar_row['actual'] - calendar_row['forecast'])
				feature2 = (calendar_row['actual'] - calendar_row['previous'])

				X_last = [feature1, feature2, future_prices[time_offset] - future_prices[0]]

				for barrier_index in range(1, 21):

					if barrier_index not in y_train_map:
						y_train_map[barrier_index] = []
						X_train_map[barrier_index] = []
						instance_offset[barrier_index] = []
						instance_offset[barrier_index].append(len(y_train_map[barrier_index]))

					start_price = future_prices[time_offset]
					top_barrier = start_price + (pip_size + (pip_size * barrier_index))
					bottom_barrier = start_price - (pip_size + (pip_size * barrier_index))

					for price in future_prices[time_offset:]:
						
						if price >= top_barrier:
							y_train_map[barrier_index].append(True)
							X_train_map[barrier_index].append(X_last)
							break

						if price <= bottom_barrier:
							y_train_map[barrier_index].append(False)
							X_train_map[barrier_index].append(X_last)
							break

			for barrier_index in instance_offset:
				instance_offset[barrier_index].append(len(y_train_map[barrier_index]))

		select_indexes = range(len(y_train))

		barrier_models = {}
		barrier_model_scores = {}
		for barrier_index in range(1, 21):

			if barrier_index not in y_train_map:
				continue

			if len(y_train_map[barrier_index]) < 50:
				continue

			cross_val_num = 10

			score = cross_val_calculator(instance_offset[barrier_index], X_train_map[barrier_index], y_train_map[barrier_index], cross_val_num, hour_lag) 

			while cross_val_num > 4:
				try:
					score = cross_val_calculator(instance_offset[barrier_index], X_train_map[barrier_index], y_train_map[barrier_index], cross_val_num, hour_lag)
					break
				except:
					cross_val_num -= 1

			if cross_val_num <= 4:
				continue

			if score > 0.5:
				auc_scores.append(score)

			barrier_model_scores[barrier_index] = score

			if barrier_index not in avg_barrier_auc:
				avg_barrier_auc[barrier_index] = []

			avg_barrier_auc[barrier_index].append(score)

			barrier_clf = xgb.XGBClassifier(seed=1)
			barrier_clf.fit(np.array(X_train_map[barrier_index]), y_train_map[barrier_index])
			barrier_models[barrier_index] = barrier_clf

		minte_lag = int(round(time_lag * 60))

		print "Mean Score", np.mean(auc_scores)

		prices = price_df[(price_df["times"] <= curr_time) & (price_df["times"] >= row["time"])]["prices"].values.tolist()

		if len(prices) == 0:
			continue

		feature1 = (row['actual'] - row['forecast'])
		feature2 = (row['actual'] - row['previous'])

		X_last = [feature1, feature2, prices[-1] - prices[0]]

		end_time = time.time()

		barriers = []
		for barrier_index in range(1, 21):
			if barrier_index not in barrier_models:
				continue

			prob = barrier_models[barrier_index].predict_proba([X_last])[0][1]
			print ("Barrier", barrier_index, prob, barrier_model_scores[barrier_index], row["currency"])

			barriers.append({"barrier" : (5 + (5 * barrier_index)), "prob" : prob, "auc" : barrier_model_scores[barrier_index]})


			
			try:
				cursor = cnx.cursor()
				query = ("INSERT INTO historic_news_barrier_probs values ( \
					'" + row["currency"] + "', \
					'" + select_pair + "', \
					'" + str(curr_time) + "', \
					'" + str(row["time"]) + "', \
					'" + str(5 + (5 * barrier_index)) + "', \
					'" + str(prob) + "', \
					'" + str(barrier_model_scores[barrier_index]) + "', \
					'" + row["description"] + "', \
					'BT1', \
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

checkIfProcessRunning('analyze_historic_news_lags_signals2.py', select_pair)

curr_time = calendar_df["time"].head(1).values.tolist()[0]
end_time = calendar_df["time"].tail(1).values.tolist()[0]

cursor = cnx.cursor()
query = ("SELECT max(time_stamp) FROM historic_news_barrier_probs where \
					currency_pair = '" + select_pair + "' and model_key = 'BT1' \
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



