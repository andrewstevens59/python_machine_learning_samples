import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from pytz import timezone
import xgboost as xgb
from bayes_opt import BayesianOptimization
from datetime import timedelta
import mysql.connector
from lxml.html import fromstring
from itertools import cycle
import gzip
import cPickle
import re

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
import pycurl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import download_calendar as download_calendar
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json


import os


import paramiko
import json

import logging
import os
import enum

def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)
    parser = fromstring(response.text)
    proxies = set()
    for i in parser.xpath('//tbody/tr')[:10]:
        if i.xpath('.//td[7][contains(text(),"yes")]'):
            #Grabbing IP and corresponding PORT
            proxy = ":".join([i.xpath('.//td[1]/text()')[0], i.xpath('.//td[2]/text()')[0]])
            proxies.add(proxy)
    return proxies

def get_calendar_day(curr_date):

	pattern = re.compile("(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)")

	from_zone = tz.gettz('US/Eastern')
	to_zone = tz.tzutc()

	url='https://www.forexfactory.com/calendar.php?day=' + curr_date
	print url

	import urllib3
	http = urllib3.PoolManager()
	response = http.request('GET', url)

	#Store the contents of the website under doc
	doc = lh.fromstring(response.data)
	#Parse data that are stored between <tr>..</tr> of HTML
	tr_elements = doc.xpath('//tr')

	currencies = ["GBP", "USD", "AUD", "CAD", "NZD", "JPY", "CHF", "EUR"]

	calendar_data = []

	curr_time = None
	#Since out first row is the header, data is stored on the second row onwards
	for j in range(0,len(tr_elements)):
		#T is our j'th row
		T=tr_elements[j]

		found_currency = False
		found_description = False

		actual = None
		forecast = None
		previous = None
		space = None
		space1 = None
		currency = None
		description = None
		timestamp = None

		#Iterate through each element of the row
		for t in T.iterchildren():
			data=t.text_content().strip()

			if found_currency == True and space1 == None:
				space1 = data
				continue

			if found_currency == True:
				found_currency = False
				found_description = True
				description = data

				continue

			if found_description == True:

				if space == None:
					space = data
					print data, "Space"
					continue

				if actual == None:
					actual = data
					print data, "Actual"
					continue

				if forecast == None:
					forecast = data
					print data, "Forecast"
					continue

				if previous == None:
					previous = data
					print previous, "Previous"
					print description, "description"
		
					try:
						non_decimal = re.compile(r'[^\d.]+')
						if len(actual) == 0:
							continue
				
						actual = float(non_decimal.sub('', actual))

						forecast = non_decimal.sub('', forecast)
						if len(forecast) > 0:
							forecast = float(forecast)
						else:
							forecast = actual

						previous = non_decimal.sub('', previous)
						if len(previous) > 0:
							previous = float(previous)
						else:
							previous = actual

						calendar_data.append([timestamp, currency, description, actual, forecast, previous]) 
					except:
						pass

					continue

			if data == "All Day":
				break

			if pattern.match(data):
				curr_time = data

				

			if data in currencies:
				print curr_date, curr_time, data
				found_currency = True
				currency = data

				local = datetime.datetime.strptime(curr_date + " " + curr_time, "%b%d.%Y %I:%M%p")

				local = local.replace(tzinfo=from_zone)

				# Convert time zone
				utc = local.astimezone(to_zone)

				timestamp = calendar.timegm(utc.timetuple())


	return calendar_data

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

		if item['lowMid'] != item['highMid'] or item['volume'] > 0:
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

			if high != low or volume > 0:
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes

def calculate_time_diff(now_time, ts):

	date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

	date = datetime.datetime.utcfromtimestamp(now_time).strftime('%Y-%m-%d %H:%M:%S')
	e = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

	_diff = (e - s)

	while s < e:
		max_hour = 24
		if s.day == e.day:
			max_hour = e.hour

		if s.weekday() in {4}:
			max_hour = 21

		if s.weekday() in {4} and s.hour in {21, 22, 23}:
			hours = 1
			_diff -= timedelta(hours=hours)
		elif s.weekday() in {5}:
			hours = max_hour - s.hour
			_diff -= timedelta(hours=hours)
		elif s.weekday() in {6} and s.hour < 21:
			hours = min(21, max_hour) - s.hour
			_diff -= timedelta(hours=hours)
		else:
			hours = max_hour - s.hour

		if hours == 0:
			break
		s += timedelta(hours=hours)

	return (_diff.total_seconds() / (60 * 60))



def cross_val_calculator(X, y, cross_val_num, is_sample_wt, params = None):

	y_true_indexes = [index for index in range(len(y)) if y[index] == True]
	y_false_indexes = [index for index in range(len(y)) if y[index] == False]

	y_test_all = []
	y_preds_all = []
	for iteration in range(cross_val_num):

		rand.seed(iteration)
		rand.shuffle(y_true_indexes)

		rand.seed(iteration)
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

		if params == None:
			clf = xgb.XGBClassifier()
		else:
			clf = xgb.XGBClassifier(
		        max_depth=int(round(params["max_depth"])),
		        learning_rate=float(params["learning_rate"]),
		        n_estimators=int(params["n_estimators"]),
		        gamma=params["gamma"])

		if is_sample_wt:
			
			true_wt = float(sum(y_train)) / len(y_train)
			false_wt = 1 - true_wt

			weights = []
			for y_s in y_train:
				if y_s:
					weights.append(false_wt)
				else:
					weights.append(true_wt)

			clf.fit(np.array(X_train), y_train, sample_weight=np.array(weights))
		else:
			clf.fit(np.array(X_train), y_train)

		preds = clf.predict_proba(np.array(X_test))[:,1]

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

import datetime as dt

class MyFormatter(logging.Formatter):
    converter=dt.datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

formatter = MyFormatter(fmt='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger



if get_mac() != 150538578859218:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 


class ModelType(enum.Enum): 
    barrier = 1
    time_regression = 2
    time_classification = 3


trade_logger = setup_logger('first_logger', root_dir + "update_news_release_signals_all" + sys.argv[1].replace(" ", "_") + ".log")

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

	from_zone = tz.gettz('US/Eastern')
	to_zone = tz.gettz('UTC')

	contents = []

	for line in lines:
		line = line[len("2018-12-23 22:44:55 "):]
		toks = line.split(",")

		if currencies == None or toks[1] in currencies:

			time = int(toks[0])

			non_decimal = re.compile(r'[^\d.]+')

			try:
				actual = float(non_decimal.sub('', toks[3]))

				forecast = non_decimal.sub('', toks[4])
				if len(forecast) > 0:
					forecast = float(forecast)
				else:
					forecast = actual

				previous = non_decimal.sub('', toks[5])
				if len(previous) > 0:
					previous = float(previous)
				else:
					previous = actual

				contents.append([toks[1], time, toks[2], actual, forecast, previous, int(toks[6])])
			except:
				pass

	return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "impact"])


df_set = []
for year in range(2007, 2020):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

def correlation_feature(before_price_df):

	X_last = []
	for price_index in range(28):
		price_mean = before_price_df['prices' + str(price_index)].mean()
		curr_price = before_price_df['prices' + str(price_index)].tail(1).iloc[0]
		X_last.append(curr_price - price_mean)

	return X_last

def linreg(X, Y):
    """
    return a,b in solution to y = ax + b such that root mean square distance between trend line and original points is minimized
    """
    N = len(X)
    Sx = Sy = Sxx = Syy = Sxy = 0.0
    for x, y in zip(X, Y):
        Sx = Sx + x
        Sy = Sy + y
        Sxx = Sxx + x*x
        Syy = Syy + y*y
        Sxy = Sxy + x*y
    det = Sxx * N - Sx * Sx
    return (Sxy * N - Sy * Sx)/det

news_release_stat_df = pd.read_csv("news_dist_stats.csv")
news_release_stat_df.set_index("key", inplace=True)

def news_movement_feature_old(before_price_df, curr_time, curr_release, delta):

	currency_map_std = {}
	currency_map_mean = {}

	curr_calendar_df = calendar_df[(calendar_df["time"] < curr_time) & (calendar_df["time"] > curr_time - (24 * 60 * 60))]

	before_prices1 = before_price_df.tail(24)
	before_prices2 = before_price_df.tail(48)
	for price_index, pair in enumerate(currency_pairs):
		price_mean = before_prices1['prices' + str(price_index)].mean()
		price_std = before_prices1['prices' + str(price_index)].std()
		curr_price = before_prices1['prices' + str(price_index)].tail(1).iloc[0]

		v1 = curr_price - price_mean
		v2 = (curr_price - price_mean) / price_std
		
		currency1 = pair[0:3]
		currency2 = pair[4:7]

		if currency1 not in currency_map_std:
			currency_map_std[currency1] = []
			currency_map_mean[currency1] = []

		if currency2 not in currency_map_std:
			currency_map_std[currency2] = []
			currency_map_mean[currency2] = []

		currency_map_std[currency1].append(v2)
		currency_map_mean[currency1].append(v1)

		currency_map_std[currency2].append(-v2)
		currency_map_mean[currency2].append(-v1)

	currency_news1_count = {}
	currency_news2_count = {}
	currency_news3_count = {}
	for index, row in curr_calendar_df.iterrows():

		time_lag = (curr_time - row["time"]) / (60 * 60) 
		currency = row["currency"]
		impact = row["impact"]

		if row["actual"] > row["forecast"]:
			diff1 = 1
		elif row["actual"] < row["forecast"]:
			diff1 = -1
		else:
			diff1 = 0

		if row["actual"] > row["previous"]:
			diff2 = 1
		elif row["actual"] < row["previous"]:
			diff2 = -1
		else:
			diff2 = 0

		if currency not in currency_news1_count:
			currency_news1_count[currency] = {}
			currency_news2_count[currency] = {}
			currency_news3_count[currency] = {}

		for impact_index in [0, impact]:
			if impact_index not in currency_news1_count[currency]:
				currency_news1_count[currency][impact_index] = [0]
				currency_news2_count[currency][impact_index] = [0]
				currency_news3_count[currency][impact_index] = [0]

			currency_news1_count[currency][impact_index][0] += diff1
			currency_news2_count[currency][impact_index][0] += diff2
			currency_news3_count[currency][impact_index][0] += 1

	X_last = [delta]
	for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:
		X_last.append(np.mean(currency_map_std[currency]))
		X_last.append(np.mean(currency_map_mean[currency]))

		if curr_release["currency"] == currency:
			X_last.append(curr_release["actual"] - curr_release["previous"])
			X_last.append(curr_release["actual"] - curr_release["forecast"])
		else:
			X_last.append(0)
			X_last.append(0)


		for impact in range(0, 4):
			if currency in currency_news1_count and impact in currency_news1_count[currency]:
				X_last += currency_news1_count[currency][impact]
				X_last += currency_news2_count[currency][impact]
				X_last += currency_news3_count[currency][impact]
			else:
				X_last += [0] * 1
				X_last += [0] * 1
				X_last += [0] * 1

	for price_index, pair in enumerate(currency_pairs):
		prices = before_prices1['prices' + str(price_index)].values.tolist()
		X_last.append(linreg([x for x in range(len(prices))], prices))

	return X_last

def news_movement_feature(before_price_df, curr_time, curr_release, delta, std_before, is_specific_model = True, is_features_only = False):


	if is_specific_model:
		X_last = [delta]
		features = ["delta"]
	else:
		X_last = []
		features = []

	currency_map_std = {}
	currency_map_mean = {}

	currency_news1_count = {}
	currency_news2_count = {}
	currency_news3_count = {}
	
	z_score_dists1 = {}
	z_score_dists2 = {}

	if is_features_only == False:
		curr_calendar_df = calendar_df[(calendar_df["time"] < curr_time) & (calendar_df["time"] > curr_time - (60 * 60 * 24 * 6))]
		
		before_prices1 = before_price_df.tail(24 * 4)
		before_prices2 = before_price_df.tail(24 * 4)
		for price_index, pair in enumerate(currency_pairs):
			price_mean = before_prices1['prices' + str(price_index)].mean()
			price_std = before_prices1['prices' + str(price_index)].std()
			curr_price = before_prices1['prices' + str(price_index)].tail(1).iloc[0]

			v1 = curr_price - price_mean
			v2 = (curr_price - price_mean) / price_std
			
			currency1 = pair[0:3]
			currency2 = pair[4:7]

			if currency1 not in currency_map_std:
				currency_map_std[currency1] = []
				currency_map_mean[currency1] = []

			if currency2 not in currency_map_std:
				currency_map_std[currency2] = []
				currency_map_mean[currency2] = []

			currency_map_std[currency1].append(v2)
			currency_map_mean[currency1].append(v1)

			currency_map_std[currency2].append(-v2)
			currency_map_mean[currency2].append(-v1)


		for index, row in curr_calendar_df.iterrows():

			time_lag = calculate_time_diff(curr_time, row["time"])

			if time_lag > 24 * 4:
				continue

			key = row["description"] + "_" + row["currency"]
			stat_row = news_release_stat_df[news_release_stat_df.index == key].iloc[0]

			sign = stat_row["sign"]

			if stat_row["forecast_std"] > 0:
				z_score1 = (float(row["actual"] - row["forecast"]) - stat_row["forecast_mean"]) / stat_row["forecast_std"]
			else:
				z_score1 = None

			if stat_row["previous_std"] > 0:
				z_score2 = (float(row["actual"] - row["previous"]) - stat_row["previous_mean"]) / stat_row["previous_std"]
			else:
				z_score2 = None

			time_lag = (curr_time - row["time"]) / (60 * 60) 
			currency = row["currency"]
			impact = row["impact"]
			

			if row["actual"] > row["forecast"]:
				diff1 = sign
			elif row["actual"] < row["forecast"]:
				diff1 = -sign
			else:
				diff1 = 0

			if row["actual"] > row["previous"]:
				diff2 = sign
			elif row["actual"] < row["previous"]:
				diff2 = -sign
			else:
				diff2 = 0

			if currency not in currency_news1_count:
				currency_news1_count[currency] = {}
				currency_news2_count[currency] = {}
				currency_news3_count[currency] = {}
				z_score_dists1[currency] = {}
				z_score_dists2[currency] = {}

			for impact_index in [0, impact]:
				if impact_index not in currency_news1_count[currency]:
					currency_news1_count[currency][impact_index] = [0]
					currency_news2_count[currency][impact_index] = [0]
					currency_news3_count[currency][impact_index] = [0]
					z_score_dists1[currency][impact_index] = []
					z_score_dists2[currency][impact_index] = []

				currency_news1_count[currency][impact_index][0] += diff1
				currency_news2_count[currency][impact_index][0] += diff2
				currency_news3_count[currency][impact_index][0] += 1

				if z_score1 != None:
					z_score_dists1[currency][impact_index].append(z_score1 * sign)

				if z_score2 != None:
					z_score_dists2[currency][impact_index].append(z_score2 * sign)

	for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:

		if is_features_only == False:
			X_last.append(np.mean(currency_map_std[currency]))
			X_last.append(np.mean(currency_map_mean[currency]))

		features.append("currency_map_std_" + currency)
		features.append("currency_map_mean_" + currency)

		if is_specific_model:

			if is_features_only == False:
				if curr_release["currency"] == currency:
					X_last.append(curr_release["actual"] - curr_release["previous"])
					X_last.append(curr_release["actual"] - curr_release["forecast"])
				else:
					X_last.append(0)
					X_last.append(0)

			features.append("actual_previous_" + currency)
			features.append("actual_forecast_" + currency)


		for impact in range(0, 4):

			if is_features_only == False:
				if currency in currency_news1_count and impact in currency_news1_count[currency]:
					X_last += currency_news1_count[currency][impact]
					X_last += currency_news2_count[currency][impact]
					X_last += currency_news3_count[currency][impact]

					dist1 = z_score_dists1[currency][impact]
					dist2 = z_score_dists2[currency][impact]

					X_last += list(np.histogram(dist1, bins=[-2000, -2, -1, 0, 1, 2, 2000])[0])
					X_last += list(np.histogram(dist2, bins=[-2000, -2, -1, 0, 1, 2, 2000])[0])
				else:
					X_last += [0] * 1
					X_last += [0] * 1
					X_last += [0] * 1

					X_last += [0] * 6
					X_last += [0] * 6

			features.append("news_count_a_f_" + currency + "_impact_" + str(impact))
			features.append("news_count_a_p_" + currency + "_impact_" + str(impact))
			features.append("news_count_t_" + currency + "_impact_" + str(impact))
			for k in range(6):
				features.append("z_score_a_f_" + currency + "_impact_" + str(impact) + "_" + str(k))
			for k in range(6):
				features.append("z_score_a_p_" + currency + "_impact_" + str(impact) + "_" + str(k))

	for price_index, pair in enumerate(currency_pairs):
		if is_features_only == False:
			prices = before_prices1['prices' + str(price_index)].values.tolist()
			X_last.append(linreg([x for x in range(len(prices))], prices))

		features.append("trend_" + pair)


	return X_last, features

def news_movement_feature1(before_price_df, curr_time, curr_release, delta):

	X_last = [delta]

	before_prices1 = before_price_df.tail(24)
	prices = before_prices1['prices_target'].values.tolist()

	X_last.append(curr_release["actual"] - curr_release["forecast"])
	X_last.append(curr_release["actual"] - curr_release["previous"])


	for price_index, pair in enumerate(currency_pairs):
		prices = before_prices1['prices' + str(price_index)].values.tolist()
		X_last.append(linreg([x for x in range(len(prices))], prices))


	currency_map_std = {}
	currency_map_mean = {}
	for price_index, pair in enumerate(currency_pairs):
		price_mean = before_prices1['prices' + str(price_index)].mean()
		price_std = before_prices1['prices' + str(price_index)].std()
		curr_price = before_prices1['prices' + str(price_index)].tail(1).iloc[0]

		v1 = curr_price - price_mean
		v2 = (curr_price - price_mean) / price_std
		
		currency1 = pair[0:3]
		currency2 = pair[4:7]

		if currency1 not in currency_map_std:
			currency_map_std[currency1] = []
			currency_map_mean[currency1] = []

		if currency2 not in currency_map_std:
			currency_map_std[currency2] = []
			currency_map_mean[currency2] = []

		currency_map_std[currency1].append(v2)
		currency_map_mean[currency1].append(v1)

		currency_map_std[currency2].append(-v2)
		currency_map_mean[currency2].append(-v1)

	for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:
		X_last.append(np.mean(currency_map_std[currency]))
		X_last.append(np.mean(currency_map_mean[currency]))


	return X_last

def get_xgb_imp(b1, feat_names):
    from numpy import array
    imp_vals = b1.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}

def create_backtest_model(select_pair, pip_size, price_df, barrier_models, barrier_model_scores, X_train_map, y_train_map, target_barrier = None):


	print ("starting")
	if os.path.isfile("models/news_barrier_model_" + select_pair + ".gz") == False:
		descriptions = list(calendar_df["description"].unique())

		train_years = []
		train_timestamps = []
		train_prices = []
		all_returns = []

		last_signal_time = 0
		barrier_pip_size = pip_size * 5
		prev_release_time = 0
		for index, row in calendar_df.iterrows():

			prev_release_time = row['time']
			curr_time = row['time']
			
			if curr_time - last_signal_time < 60 * 60 * 6:
				continue

			last_signal_time = curr_time
			year = int(datetime.datetime.utcfromtimestamp(curr_time).strftime('%Y'))

			before_price_df = price_df[price_df['times'] < curr_time]
			if len(before_price_df) < 24:
				continue

			future_price_df = price_df[price_df['times'] >= curr_time]
			future_prices = (future_price_df)['prices_target'].values.tolist()
			start_price = future_prices[0]

			X_last, _ = news_movement_feature(before_price_df, curr_time, row, 0, 0, is_specific_model=False)
			if 0 not in X_train_map:
				X_train_map[0] = []

			X_train_map[0].append(X_last)
			train_years.append(year)
			train_timestamps.append(curr_time)
			train_prices.append(start_price)
			for barrier_index in range(1, 21):

				if barrier_index not in y_train_map:
					y_train_map[barrier_index] = []
					X_train_map[barrier_index] = []

				top_barrier = start_price + (barrier_pip_size + (barrier_pip_size * barrier_index))
				bottom_barrier = start_price - (barrier_pip_size + (barrier_pip_size * barrier_index))

				found = False
				for price in future_prices:
					
					if price >= top_barrier:
						y_train_map[barrier_index].append(True)
						found = True
						break

					if price <= bottom_barrier:
						y_train_map[barrier_index].append(False)
						found = True
						break

				if found == False:
					y_train_map[barrier_index].append(price > start_price)

		models = {"X" : X_train_map, "y" : y_train_map, "train_years" : train_years,  "train_timestamps" : train_timestamps, "train_prices" : train_prices}
		fp=gzip.open("models/news_barrier_model_" + select_pair + ".gz", "wb")
		cPickle.dump(models,fp)
		fp.close()
	else:
		print ("loading data")
		fp=gzip.open("models/news_barrier_model_" + select_pair + ".gz", "rb")
		models = cPickle.load(fp)
		fp.close()

		X_train_map = models["X"]
		y_train_map = models["y"]
		train_years = models["train_years"]
		train_timestamps = models["train_timestamps"]
		train_prices = models["train_prices"]
				

	if os.path.isfile("models/historic_predictions_" + select_pair + ".gz") == False:
		data  = {}
	else:
		fp=gzip.open("models/historic_predictions_" + select_pair + ".gz", "rb")
		data = cPickle.load(fp)
		fp.close()

	for select_year in range(2007, 2020):

		print (select_year)

		if select_year not in data:
			data[select_year] = {}

		for barrier_index in y_train_map:
			print (barrier_index)

			if barrier_index in data[select_year]:
				continue

			prices = train_prices[:len(y_train_map[barrier_index])]
			timestamps = train_timestamps[:len(y_train_map[barrier_index])]
			years = train_years[:len(y_train_map[barrier_index])]
			X = X_train_map[0][:len(y_train_map[barrier_index])]
			y = y_train_map[barrier_index]

			X_test = [x for x, year in zip(X, years) if year == select_year]
			timestamps = [t for t, year in zip(timestamps, years) if year == select_year]
			prices = [p for p, year in zip(prices, years) if year == select_year]

			X = [x for x, year in zip(X, years) if year != select_year]
			y = [y1 for y1, year in zip(y, years) if year != select_year]

			clf = xgb.XGBClassifier()
			clf.fit(np.array(X), y)
			predictions = clf.predict_proba(np.array(X_test))[:,1]

			data[select_year][barrier_index] = {}
			data[select_year][barrier_index]["timestamp"] = timestamps
			data[select_year][barrier_index]["predictions"] = predictions
			data[select_year][barrier_index]["prices"] = prices

			fp=gzip.open("models/historic_predictions_" + select_pair + ".gz", "wb")
			cPickle.dump(data,fp)
			fp.close()


def train_barrier_model(pip_size, price_df, barrier_models, barrier_model_scores, X_train_map, y_train_map, target_barrier = None):

	descriptions = list(calendar_df["description"].unique())

	auc_all = {3 : [], 6 : [], 12 : [], 24 : [], 48 : []}
	for description in descriptions:
		for impact in [1, 2, 3]:

			X_train_map = {}
			y_train_map = {}

			sub_df = calendar_df[(calendar_df["description"] == description) & (calendar_df["impact"] == impact)]
			if len(sub_df) < 40:
				continue

			print (description, impact, len(sub_df))

			prev_release_time = 0
			for hour_afters in [3, 6, 12, 24, 48]:
				for index, row in sub_df.iterrows():

					if abs(row['time'] - prev_release_time) < 48 * 60 * 60:
						continue

					prev_release_time = row['time']
					curr_time = row['time'] + (hour_afters * 60 * 60)

					before_price_df = price_df[price_df['times'] < curr_time]
					if len(before_price_df) < 24:
						continue

					release_price = price_df['prices_target'][price_df['times'] >= row['time']].head(1).values.tolist()[0]

					future_price_df = price_df[price_df['times'] >= curr_time]
					future_prices = (future_price_df)['prices_target'].values.tolist()

					start_price = future_prices[0]
					delta = start_price - release_price

					#X_last = correlation_feature(before_price_df)
					X_last = news_movement_feature(before_price_df, curr_time, row, delta)

					for barrier_index in range(1, 21):

						if barrier_index in barrier_models:
							continue

						if barrier_index not in y_train_map:
							y_train_map[barrier_index] = []
							X_train_map[barrier_index] = []

						if model_type == ModelType.barrier:
							top_barrier = start_price + (pip_size + (pip_size * barrier_index))
							bottom_barrier = start_price - (pip_size + (pip_size * barrier_index))

							for price in future_prices:
								
								if price >= top_barrier:
									y_train_map[barrier_index].append(True)
									X_train_map[barrier_index].append(X_last)
									break

								if price <= bottom_barrier:
									y_train_map[barrier_index].append(False)
									X_train_map[barrier_index].append(X_last)
									break

				'''
				pickle.dump(y_train_map, open("y_train.pickle", "w"))
				pickle.dump(X_train_map, open("X_train.pickle", "w"))
				

				X_train_map = pickle.load(open("X_train.pickle", "r"))
				y_train_map = pickle.load(open("y_train.pickle", "r"))
			
				'''

				feat_names = []
				for currency in ["EUR", "USD", "GBP", "CAD", "NZD", "AUD", "JPY", "CHF"]:
					feat_names.append(currency + "_std")
					feat_names.append(currency + "_mean")
					feat_names += [currency + "_news_count_" + str(i) for i in range(5)]


				for barrier_index in y_train_map:

					if barrier_index in barrier_models:
						continue

					if barrier_index not in y_train_map:
						continue

					if len(y_train_map[barrier_index]) < 30:
						continue

					X = X_train_map[barrier_index]
					y = y_train_map[barrier_index]

					start = 0
					end = len(X_train_map[barrier_index]) / 10

					probs_all = []
					y_all = []
					for i in range(10):
						clf = xgb.XGBClassifier()
						clf.fit(np.array(X[:start] + X[end:]), y[:start] + y[end:])

						#print ([[a, b] for a, b in zip(feat_names, clf.feature_importances_)])


						probs = clf.predict_proba(np.array(X[start:end]))[:,1]

						probs_all += list(probs)
						y_all += list(y[start:end])

						start += len(X_train_map[barrier_index]) / 10
						end += len(X_train_map[barrier_index]) / 10

					fpr, tpr, thresholds = metrics.roc_curve(y_all, probs_all)
					score = metrics.auc(fpr, tpr)

					barrier_model_scores[barrier_index] = score

					auc_all[hour_afters].append(score)

					print (hour_afters, barrier_index, score, np.percentile(auc_all[hour_afters], 90))


	return barrier_models, barrier_model_scores, X_train_map, y_train_map


def back_test_news_calendar(select_pairs, model_type):

	news_summary = []
	stat_dict = {}
	diff = 0

	for currency_pair in currency_pairs:

		print (currency_pair)

		prices, times, volumes = load_time_series(currency_pair, None, True)
		price_df = pd.DataFrame()
		price_df["prices_target"] = prices
		price_df["times"] = times

		for i, compare_pair in enumerate(currency_pairs):
			prices, times, volumes = load_time_series(compare_pair, None, True)
			price_df2 = pd.DataFrame()
			price_df2["prices" + str(i)] = prices
			price_df2["times"] = times

			price_df = price_df.set_index('times').join(price_df2.set_index('times'), how='inner')
			price_df.reset_index(inplace=True)

		if currency_pair[4:7] == "JPY":
			pip_size = 0.01
		else:
			pip_size = 0.0001

		start_time = time.time()



		create_backtest_model(currency_pair, pip_size, price_df, {}, {}, {}, {})
		#shapley_importance(currency_pair)

	return stat_dict

import psutil

def checkIfProcessRunning(processName, command):
	count = 0
	#Iterate over the all the running process
	for proc in psutil.process_iter():

		try:
			cmdline = proc.cmdline()

			# Check if process name contains the given name string.
			if len(cmdline) > 3 and processName.lower() in cmdline[2] and command in cmdline[3]:
				count += 1
			elif len(cmdline) > 2 and processName.lower() in cmdline[1] and command in cmdline[2]: 
				count += 1
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			pass

	if count >= 2:
		sys.exit(0)


checkIfProcessRunning('execute_all_update_news_signals.py', sys.argv[1])

model_type = sys.argv[2]
if model_type == "barrier":
	model_type = ModelType.barrier
elif model_type == "time_regression":
	model_type = ModelType.time_regression
elif model_type == "time_classification":
	model_type = ModelType.time_classification

def process_demo_pairs():

	pairs = sys.argv[1].split(",")
	stat_dict = back_test_news_calendar(pairs, model_type)

	trade_logger.info('Finished ' + str(stat_dict.keys())) 

process_demo_pairs()

