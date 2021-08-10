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
import sysc
import re

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import download_calendar as download_calendar
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json


import os
from maximize_sharpe import *


import paramiko
import json

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
		with open("/Users/callummc/Documents/economic_calendar/calendar_" + str(year) + ".txt") as f:
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


def load_time_series(symbol, year):

	if get_mac() == 154505288144005:
		prefix = '/Users/callummc/'
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


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


def get_lock(process_name):
    # Without holding a reference to our socket somewhere it gets garbage
    # collected when the function exits
    get_lock._lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    try:
        get_lock._lock_socket.bind('\0' + process_name)
    except socket.error:
    	print "Already Running"
        sys.exit()


#get_lock('historic_news_movement')


curr_calendar_df = get_curr_calendar_day()
print curr_calendar_df

df_set = []
for year in range(2007, 2019):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

stat_dict = {}
stat_dict["new_releases"] = []

for currency_pair in currency_pairs:

	print currency_pair
	prices, times = load_time_series(currency_pair, None)

	price_df = pd.DataFrame()
	price_df['times'] = times
	price_df["prices"] = prices
	price_df['mult_factor'] = 1.0
	price_df.set_index('times', inplace=True)

	price_df.fillna(method='ffill', inplace=True)
	price_df.reset_index(inplace=True)

	X_train = []
	y_train = []

	if currency_pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	start_time = time.time()

	
	X_trends = []
	for index2, calendar_row in calendar_df.iterrows():
		future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].head(12).values.tolist()
		if len(future_prices) < 12:
			continue

		X_trends.append([future_prices[-1] - future_prices[0]])

	kmeans = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1, 
	                               random_state = 42).fit(X_trends)

	cluster_ids = kmeans.predict(X_trends)

	histogram = []

	y_train_map = {}
	X_train_map = {}

	curr_index = 0
	for index2, calendar_row in calendar_df.iterrows():
		curr_index += 1

		last_time = (price_df[price_df['times'] <= calendar_row['time']])['prices'].tail(24 * 5).values.tolist()[0]

		calendar_items = calendar_df[(calendar_df['time'] >= last_time) & (calendar_df['time'] <= calendar_row["time"])]
		if len(calendar_items) < 6:
			continue

		total_items = 0
		histogram = [0] * 20
		for cluster_index in range(max(0, curr_index - len(calendar_items)), curr_index):
			histogram[cluster_ids[cluster_index]] += 1
			total_items += 1

		histogram = [float(v) / total_items for v in histogram]

		future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()
		y_train.append(future_prices[-1] - future_prices[0])

		for barrier_index in range(1, 15):

			if barrier_index not in y_train_map:
				y_train_map[barrier_index] = []
				X_train_map[barrier_index] = []

			start_price = future_prices[12]
			top_barrier = start_price + (pip_size + (pip_size * barrier_index * 10))
			bottom_barrier = start_price - (pip_size + (pip_size * barrier_index * 10))

			for price in future_prices[12:]:
				
				if price >= top_barrier:
					y_train_map[barrier_index].append(True)
					X_train_map[barrier_index].append(histogram)
					break

				if price <= bottom_barrier:
					y_train_map[barrier_index].append(False)
					X_train_map[barrier_index].append(histogram)
					break

	barrier_models = {}
	barrier_model_scores = {}
	for barrier_index in range(1, 15):
		if len(y_train_map[barrier_index]) < 20:
			continue

		barrier_clf = GradientBoostingClassifier(random_state=42)

		cross_val_num = 10

		while True:
			try:
				scores = cross_val_score(barrier_clf, X_train_map[barrier_index], y_train_map[barrier_index], cv=cross_val_num, scoring='roc_auc')
				score = np.mean(scores)
				break
			except:
				cross_val_num -= 1

		barrier_model_scores[barrier_index] = score

		if score < 0.57:
			continue

		barrier_clf.fit(X_train_map[barrier_index], y_train_map[barrier_index])
		barrier_models[barrier_index] = barrier_clf

	total_count = 0
	histogram = [float(v) / total_items for v in histogram]
	for index, row in curr_calendar_df[(curr_calendar_df["currency"] == currency_pair[0:3]) | (curr_calendar_df["currency"] == currency_pair[4:7])].iterrows():
		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].values.tolist()
		if len(future_prices) < 12:
			continue

		cluster_id = kmeans.predict([future_prices[-1] - future_prices[0]])
		histogram[cluster_id] += 1
		total_count += 1

	if total_count > 5:
		histogram = [float(v) / total_count for v in histogram]

		barriers = []
		for barrier_index in range(1, 15):
			if barrier_index not in barrier_models:
				continue

			prob = barrier_models[barrier_index].predict_proba([histogram])[0][1]

			print "Barrier", barrier_index, prob, barrier_model_scores[barrier_index]

			barriers.append({"barrier" : (10 + (10 * barrier_index)), "prob" : prob, "auc" : barrier_model_scores[barrier_index]})

		stat_dict["new_releases"].append({"barriers" : barriers, "time_lag" : time_lag, "currency" : calendar_row["currency"], "actual" : calendar_row['actual'], "forecast" : calendar_row['forecast'], "previous" : calendar_row['previous'], "pair" : currency_pair, "description" : row["description"], "mean" : mean, "std" : std, "z-score" : z_score})


with open('/var/www/html/calendar_output.json', 'w') as fp:
    json.dump(stat_dict, fp)

