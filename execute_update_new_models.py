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

from os import listdir
from os.path import isfile, join

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
from xgboost import XGBClassifier
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
					except:
						actual = -1
						forecast = -1
						previous = -1


					calendar_data.append([timestamp, currency, description, actual, forecast, previous]) 
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

				contents.append([toks[2], time, toks[3], toks[4], toks[5], toks[6], year])
			except:
				pass

	return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "year"])


def get_curr_calendar_day():

	d = datetime.datetime.now() + datetime.timedelta(days=1)

	curr_date = datetime.datetime.now().strftime("%b%d.%Y").lower()

	week_day = datetime.datetime.now().weekday()

	print "curr day", week_day

	
	if os.path.isfile("/tmp/calendar_data"):
		calendar = pickle.load(open("/tmp/calendar_data", 'rb'))

		news_times = calendar["df"]["time"].values.tolist()

		found_recent_news = False
		for news_time in news_times:
			if abs(time.time() - news_time) < 1 * 60:
				print "find new news"
				found_recent_news = True


		if found_recent_news == False and calendar["day"] == curr_date and abs(time.time() - calendar["last_check"]) < 6 * 60 * 60:
			if len(calendar["df"]) > 0:
				return calendar["df"]
	
	print "loading..."
	day_after = d.strftime("%b%d.%Y").lower()

	print "curr date", curr_date
	print "day after", day_after

	calendar_data = get_calendar_day(curr_date) + get_calendar_day(day_after)


	calendar = {"last_check" : time.time(), "day" :  curr_date, "df" : pd.DataFrame(calendar_data, columns=["time", "currency", "description", "actual", "forecast", "previous"])}

	pickle.dump(calendar, open("/tmp/calendar_data", 'wb'))
	return calendar["df"]


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

			clf_mean = GradientBoostingRegressor(min_samples_leaf=5, max_depth=30, n_estimators=150)
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

def back_test_news_model(test_calendar, price_df, year, pair):

	avg_spreads = pickle.load(open(root_dir + "pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
		pip_mult = 100.0
	else:
		pip_size = 0.0001
		pip_mult = 1.0

	commission = avg_spreads[pair] * pip_size

	model_map = {}
	y_train_map = {}
	X_train = []

	X_test = []
	y_test_map = {}
	for index2, calendar_row in test_calendar.iterrows():

		future_prices = (price_df[price_df['times'] >= calendar_row['time']])['prices'].values.tolist()
		future_times = (price_df[price_df['times'] >= calendar_row['time']])['times'].values.tolist()

		for time_index in range(12, 24):

			start_price = future_prices[time_index]
			start_time = future_times[time_index]

			for barrier_index in range(1, 15):

				if barrier_index not in y_train_map:
					y_train_map[barrier_index] = []
					y_test_map[barrier_index] = []

				top_barrier = start_price + ((0.001 * pip_mult) + (0.001 * barrier_index * pip_mult))
				bottom_barrier = start_price - ((0.001 * pip_mult) + (0.001 * barrier_index * pip_mult))

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
				feature1 = (calendar_row['actual'] - calendar_row['forecast'])
				feature2 = (calendar_row['actual'] - calendar_row['previous'])

				if calendar_row["currency"] == pair[0:3]:
					features = [feature1, feature2, 0, 0]
				else:
					features = [0, 0, feature1, feature2]

				z_score, model_map = find_z_score(pair, test_calendar, time_index, future_prices[time_index] - future_prices[0], features, model_map, year, price_df)
				if calendar_row["year"] == year:
					X_test.append([z_score])
				else:
					X_train.append([z_score])

			else:
				print "no"

	best_score_index = None
	best_score = 0
	for barrier_index in range(1, 15):
		try:
			clf = GradientBoostingClassifier(min_samples_leaf=5, max_depth=30, n_estimators=150)
			#clf = XGBClassifier(random_state=42)
	
			scores = cross_val_score(clf, X_train, y_train_map[barrier_index], cv=5, scoring='roc_auc')
			score = np.mean(scores)

			if score > 0.50 and score > best_score:
				best_score = score
				best_score_index = barrier_index

			print barrier_index, np.mean(scores)
		except:
			pass

	if best_score_index == None:
		return None, None, None, None, None, None

	clf = GradientBoostingClassifier(min_samples_leaf=5, max_depth=30, n_estimators=150)
	#clf = XGBClassifier(random_state=42)
	clf.fit(X_train,  y_train_map[best_score_index])

	print "test", len(y_test_map[best_score_index])

	return clf, X_test, y_test_map[best_score_index], ((0.001 * pip_mult) + (0.001 * best_score_index * pip_mult)), best_score, model_map


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


def create_model_whitelist(currency_pair, description, summary_map):

	total_profit = 0
	for year in range(2007, 2019):

		if (str(year) + "_" + currency_pair + "_" + description) in summary_map:
			total_profit += summary_map[str(year) + "_" + currency_pair + "_" + description]["total_profit"]

	if total_profit <= 0:
		return

	profit = abs(total_profit)

	test_calendar = calendar_df[(calendar_df["description"] == description)]

	test_calendar = test_calendar[(test_calendar["currency"] == currency_pair[0:3]) | (test_calendar["currency"] == currency_pair[4:7])]

	print currency_pair, description

	prices, times = load_time_series(currency_pair, None)

	price_df = pd.DataFrame()
	price_df['times'] = times
	price_df["prices"] = prices
	price_df['mult_factor'] = 1.0
	price_df.set_index('times', inplace=True)

	price_df.fillna(method='ffill', inplace=True)
	price_df.reset_index(inplace=True)
	
	clf, X_test, y_test, barrier_size, score, model_map = back_test_news_model(test_calendar, price_df, None, currency_pair)

	if clf != None:

		whitelist_map = {}
		whitelist_map["forecast_clfs"] = model_map
		whitelist_map["model"] = clf
		whitelist_map["barrier_size"] = barrier_size
		whitelist_map["order_magnitude"] = total_profit

		trade_logger.info('Saving Model: ' + currency_pair + "_" + description.replace(" ", "_").replace("/", "_")) 
		pickle.dump(whitelist_map, open(root_dir + "trading_data/news_release_meta_" + currency_pair + "_" + description.replace(" ", "_").replace("/", "_"),"wb"))


df_set = []
for year in range(2007, 2019):
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

def process_pair_description(currency_pair, description):

	summary_map = {}

	if currency_pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[currency_pair] * pip_size

	test_calendar = calendar_df[(calendar_df["description"] == description)]

	test_calendar = test_calendar[(test_calendar["currency"] == currency_pair[0:3]) | (test_calendar["currency"] == currency_pair[4:7])]

	if len(test_calendar) < 25:
		return None

	trade_logger.info('Updating Model: ' + currency_pair + "_" + description.replace(" ", "_").replace("/", "_")) 

	prices, times = load_time_series(currency_pair, None)

	price_df = pd.DataFrame()
	price_df['times'] = times
	price_df["prices"] = prices
	price_df['mult_factor'] = 1.0
	price_df.set_index('times', inplace=True)

	price_df.fillna(method='ffill', inplace=True)
	price_df.reset_index(inplace=True)

	total_profit = 0
	for year in range(2007, 2019):

		if (str(year) + "_" + currency_pair + "_" + description) in summary_map:
			continue

		if len(test_calendar[test_calendar["year"] == year]) == 0:
			continue

	clf, X_test, y_test, barrier_size, score, model_map = back_test_news_model(test_calendar, price_df, year, currency_pair)
	if clf == None:
		return None

	order_pnls = []
	for x, y in zip(X_test, y_test):
		prob = clf.predict_proba([x])[0][1]

		future_prices = (price_df[price_df['times'] >= y])['prices'].values.tolist()
		future_times = (price_df[price_df['times'] >= y])['times'].values.tolist()

		top_gap = 0
		end_gap = 0

		top_barrier = future_prices[0] + barrier_size
		bottom_barrier = future_prices[0] - barrier_size

		if prob > 0.5:
			top_gap = commission
		else:
			end_gap = commission

		pnl = None

		end_time = 0
		for price, time in zip(future_prices, future_times):

			
			if price >= top_barrier + top_gap:
				pnl = (top_barrier + top_gap) - future_prices[0]
				end_time = time
				break

			if price <= bottom_barrier - end_gap:
				pnl = (bottom_barrier - end_gap) - future_prices[0]
				end_time = time
				break

			if (prob > 0.5) == (price > future_prices[0]):
				calc_pnl = abs(price - future_prices[0])
			else:
				calc_pnl = -abs(price - future_prices[0])

			order_pnls.append([time, ((calc_pnl - commission) * abs(prob - 0.5) * score) / barrier_size])

		if pnl != None:
			if (prob > 0.5) == (pnl > 0):
				final_pnl = ((abs(pnl) - commission) * abs(prob - 0.5) * score) / barrier_size
			else:
				final_pnl = -((abs(pnl) - commission) * abs(prob - 0.5) * score) / barrier_size

		amount = (abs(prob - 0.5) * score) / barrier_size

		time_lapse = float(end_time - future_times[0])
		carry_cost = 150 * float((time_lapse / (60 * 60)) / 8760) * (float(amount) / 10000)
		total_profit += final_pnl - carry_cost


	summary_map[str(year) + "_" + currency_pair + "_" + description] = {"total_profit" : total_profit, "pnls" : [], "barrier_size" : barrier_size}

	return summary_map


if get_mac() != 154505288144005:
	avg_spreads = pickle.load(open("/root/pair_avg_spread", 'rb'))
	root_dir = "/root/" 
else:
	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	root_dir = "/tmp/" 

trade_logger = setup_logger('first_logger', root_dir + "update_news_models.log")

def get_lock(process_name):
    # Without holding a reference to our socket somewhere it gets garbage
    # collected when the function exits
    get_lock._lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)

    try:
        get_lock._lock_socket.bind('\0' + process_name)
    except socket.error:
        trade_logger.info("Another Instance Running Still ...")
        sys.exit()


get_lock('news_release_update_model')

curr_day_calendar_df = get_curr_calendar_day()


def cleanup_old_files(calendar, model_calendar_processed_set):

	mypath = root_dir + "trading_data"

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	for file in onlyfiles:

		if "news_release_meta_" not in file:
			continue

		if ".gz" in file:
			file_match = file[:-3]
		else:
			file_match = file

		found = False
		for index, row in calendar.iterrows():

			for pair in currency_pairs:
				news_time = row["time"]
				model_key = pair + "_" + row["description"].replace(" ", "_").replace("/", "_")

				if file_match == ("news_release_meta_" + model_key):
					found = True
					break

			if found:
				break

		if found == False:
			os.remove(root_dir + "trading_data/" + file)
			trade_logger.info("Removing: " + root_dir + "trading_data/" + file)

	remove_keys = []
	for model_key in model_calendar_processed_set:

		found = False
		for index, row in calendar.iterrows():

			for pair in currency_pairs:
				news_time = row["time"]
				compare_model_key = pair + "_" + row["description"].replace(" ", "_").replace("/", "_")

				if compare_model_key == model_key:
					found = True
					break

			if found:
				break

		if found == False:
			remove_keys.append(model_key)
			trade_logger.info("Removing: " + model_key)

	for model_key in remove_keys:
		model_calendar_processed_set.remove(model_key)

	if len(model_calendar_processed_set) > 0:
		pickle.dump(model_calendar_processed_set, open(root_dir + "model_calendar_processed", "wb"))

	return model_calendar_processed_set


try:
	model_calendar_processed_set = pickle.load(open(root_dir + "model_calendar_processed", "rb"))
except:
	model_calendar_processed_set = set()

model_calendar_processed_set = cleanup_old_files(curr_day_calendar_df, model_calendar_processed_set)

for index, row in curr_day_calendar_df.iterrows():

	for currency_pair in currency_pairs:

		if currency_pair[0:3] != row["currency"] and currency_pair[4:7] != row["currency"]:
			continue

		if time.time() - row["time"] < 60 * 60 * 12 and row["time"] < time.time():
			continue

		model_key = currency_pair + "_" + row["description"].replace(" ", "_").replace("/", "_")
		if model_key in model_calendar_processed_set:
			continue

		summary_map = process_pair_description(currency_pair, row["description"])
		if summary_map != None:
			create_model_whitelist(currency_pair, row["description"], summary_map)

		model_calendar_processed_set.add(model_key)

		pickle.dump(model_calendar_processed_set, open(root_dir + "model_calendar_processed", "wb"))

		

