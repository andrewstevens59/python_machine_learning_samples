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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import download_calendar as download_calendar
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json


import os
from maximize_sharpe import *


import logging
import os


def get_calendar_day(curr_date):

	pattern = re.compile("(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)")

	from_zone = tz.tzlocal()
	to_zone = tz.tzutc()

	url='https://www.forexfactory.com/calendar.php?day=' + curr_date
	print url
	#Create a handle, page, to handle the contents of the website
	page = requests.get(url)
	#Store the contents of the website under doc
	doc = lh.fromstring(page.content)
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
		
					non_decimal = re.compile(r'[^\d.]+')

					try:
						actual = float(non_decimal.sub('', actual))
						forecast = float(non_decimal.sub('', forecast))
						previous = float(non_decimal.sub('', previous))
						calendar_data.append([timestamp, currency, description, actual, forecast, previous]) 
					except:
						actual = None
						forecast = None
						previous = None

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

def get_curr_calendar_day():

	curr_date = datetime.datetime.now().strftime("%b%d.%Y").lower()

	week_day = datetime.datetime.now().weekday()

	print "curr day", week_day

	
	if os.path.isfile("/tmp/calendar_data_historic"):
		calendar = pickle.load(open("/tmp/calendar_data_historic", 'rb'))

		news_times = calendar["df"]["time"].values.tolist()

		found_recent_news = False
		for news_time in news_times:
			print abs(time.time() - news_time) 
			if abs(time.time() - news_time) < 1 * 60:
				print "find new news"
				found_recent_news = True


		if found_recent_news == False and calendar["day"] == curr_date and abs(time.time() - calendar["last_check"]) < 6 * 60 * 60:
			if len(calendar["df"]) > 0:
				return calendar["df"]
	
	print "loading..."


	calendar_data = get_calendar_day(curr_date)

	for back_day in range(1, 7):
		d = datetime.datetime.now() - datetime.timedelta(days=back_day)

		day_before = d.strftime("%b%d.%Y").lower()
		calendar_data = get_calendar_day(day_before) + calendar_data


	calendar = {"last_check" : time.time(), "day" :  curr_date, "df" : pd.DataFrame(calendar_data, columns=["time", "currency", "description", "actual", "forecast", "previous"])}

	pickle.dump(calendar, open("/tmp/calendar_data_historic", 'wb'))
	return calendar["df"]



def get_calendar_df(pair, year): 

	if pair != None:
		currencies = [pair[0:3], pair[4:7]]
	else:
		currencies = None

	if get_mac() == 154505288144005:
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

	if get_mac() == 154505288144005:
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

def calculate_time_diff(ts):

	date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

	week_day_start = date.weekday()

	time_start = calendar.timegm(date.timetuple())

	s = str(datetime.datetime.utcnow())
	s = s[0 : s.index('.')]

	week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

	time_end = time.time()

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

		clf = GradientBoostingClassifier(random_state=42)
		clf.fit(X_train, y_train)

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

import psutil
 
def checkIfProcessRunning(processName):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
			cmdline = proc.cmdline()
			# Check if process name contains the given name string.
			if len(cmdline) > 2 and processName.lower() in cmdline[2]:
				count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 2:
    	sys.exit(0)


checkIfProcessRunning('execute_historic_news_statistic.py')

curr_calendar_df = get_curr_calendar_day()
print curr_calendar_df

df_set = []
for year in range(2007, 2019):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

currency_set = set(curr_calendar_df["currency"].values.tolist())

avg_barrier_auc = {}

stat_dict = {}
stat_dict["new_releases"] = []
for currency in currency_set:

	for index, row in curr_calendar_df[curr_calendar_df["currency"] == currency].iterrows():

		print row

		time_lag = calculate_time_diff(row["time"])
		hour_lag = int(round(time_lag))
		if time_lag < 4:
			print "< 1", time_lag
			continue

		test_calendar = calendar_df[(calendar_df["description"] == row["description"]) & (calendar_df["currency"] == row["currency"])]

		for barrier_index in avg_barrier_auc:
			print "Mean AUC Barrier", barrier_index, np.mean(avg_barrier_auc[barrier_index])

		for currency_pair in currency_pairs:

			if (currency_pair[0:3] == row["currency"]) or currency_pair[4:7] == row["currency"]:
				print currency_pair
				prices, times, volumes = load_time_series(currency_pair, None, True)
				buy_price_df = pd.DataFrame()
				buy_price_df['times'] = times
				buy_price_df["price_buy"] = prices
				buy_price_df["volume_buy"] = volumes
				buy_price_df.set_index('times', inplace=True)
				buy_price_df.fillna(method='ffill', inplace=True)

				prices, times, volumes = load_time_series(currency_pair, None, False)
				sell_price_df = pd.DataFrame()
				sell_price_df['times'] = times
				sell_price_df["price_sell"] = prices
				sell_price_df["volume_sell"] = volumes
				sell_price_df.set_index('times', inplace=True)
				sell_price_df.fillna(method='ffill', inplace=True)
		
				price_df = buy_price_df.join(sell_price_df)
				price_df["prices"] = price_df.apply(lambda x: (x["price_buy"] + x["price_sell"]) * 0.5, axis=1)
				price_df.reset_index(inplace=True)

				X_train = []
				y_train = []

				if currency_pair[4:7] == "JPY":
					pip_size = 0.1
				else:
					pip_size = 0.001

				start_time = time.time()

				y_train_map = {}
				X_train_map = {}

				for index2, calendar_row in test_calendar.iterrows():
					future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].head(hour_lag).values.tolist()
					future_buy_volumes = (price_df[price_df['times'] >= calendar_row['time']])['volume_buy'].head(hour_lag).sum()
					future_sell_volumes = (price_df[price_df['times'] >= calendar_row['time']])['volume_sell'].head(hour_lag).sum()
					if len(future_prices) < 4:
						continue

					feature1 = (calendar_row['actual'] - calendar_row['forecast'])
					feature2 = (calendar_row['actual'] - calendar_row['previous'])
					feature3 = (future_buy_volumes - future_sell_volumes) / (future_buy_volumes + future_sell_volumes)

					if calendar_row["currency"] == currency_pair[0:3]:
						X_train.append([feature1, feature2, 0, 0])
						X_last = [feature1, feature2, 0, 0, future_prices[-1] - future_prices[0]]
					else:
						X_train.append([0, 0, feature1, feature2])
						X_last = [0, 0, feature1, feature2, future_prices[-1] - future_prices[0]]

					y_train.append(future_prices[-1] - future_prices[0])

					future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()

					for barrier_index in range(3, 20):

						if barrier_index not in y_train_map:
							y_train_map[barrier_index] = []
							X_train_map[barrier_index] = []

						start_price = future_prices[hour_lag]
						top_barrier = start_price + (pip_size + (pip_size * barrier_index))
						bottom_barrier = start_price - (pip_size + (pip_size * barrier_index))

						for price in future_prices[hour_lag:]:
							
							if price >= top_barrier:
								y_train_map[barrier_index].append(True)
								X_train_map[barrier_index].append(X_last)
								break

							if price <= bottom_barrier:
								y_train_map[barrier_index].append(False)
								X_train_map[barrier_index].append(X_last)
								break

				select_indexes = range(len(y_train))

				barrier_models = {}
				barrier_model_scores = {}
				for barrier_index in range(3, 20):
					if barrier_index not in y_train_map:
						continue
						
					if len(y_train_map[barrier_index]) < 50:
						continue

					cross_val_num = 10

					while cross_val_num > 4:
						try:
							score = cross_val_calculator(X_train_map[barrier_index], y_train_map[barrier_index], cross_val_num)
							break
						except:
							cross_val_num -= 1

					if cross_val_num <= 4:
						continue

					barrier_model_scores[barrier_index] = score

					if barrier_index not in avg_barrier_auc:
						avg_barrier_auc[barrier_index] = []

					avg_barrier_auc[barrier_index].append(score)

					if score < 0.57:
						continue

					barrier_clf = GradientBoostingClassifier(random_state=42)
					barrier_clf.fit(X_train_map[barrier_index], y_train_map[barrier_index])
					print "Feature Importances", barrier_clf.feature_importances_
					barrier_models[barrier_index] = barrier_clf

				minte_lag = int(round(time_lag * 60))

				if minte_lag < 5000:
					prices, times = get_time_series(currency_pair, minte_lag, granularity="M1")
				else:
					prices, times = get_time_series(currency_pair, hour_lag)

				feature1 = (row['actual'] - row['forecast'])
				feature2 = (row['actual'] - row['previous'])

				if row["currency"] == currency_pair[0:3]:
					feature_vector = [feature1, feature2, 0, 0]
					X_last = [feature1, feature2, 0, 0, prices[-1] - prices[0]]
				else:
					feature_vector = [0, 0, feature1, feature2]
					X_last = [0, 0, feature1, feature2, prices[-1] - prices[0]]

				end_time = time.time()
				print "process time", end_time - start_time

				barriers = []
				for barrier_index in range(3, 20):
					if barrier_index not in barrier_models:
						continue

					prob = barrier_models[barrier_index].predict_proba([X_last])[0][1]
					print "Barrier", barrier_index, prob

					barriers.append({"barrier" : (10 + (10 * barrier_index)), "prob" : prob, "auc" : barrier_model_scores[barrier_index]})

				stat_dict["new_releases"].append({"barriers" : barriers, "time_lag" : int(time_lag), "currency" : calendar_row["currency"], "actual" : calendar_row['actual'], "forecast" : calendar_row['forecast'], "previous" : calendar_row['previous'], "pair" : currency_pair, "description" : row["description"]})

stat_dict["last_update_time"] = time.time()

with open('/var/www/html/calendar_output.json', 'w') as fp:
    json.dump(stat_dict, fp)

