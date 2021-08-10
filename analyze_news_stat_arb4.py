


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
import re

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



def create_training_set(currency1, currency2, price_df, pair, year, test_calendar, description_map):

	X_train = []
	y_train = []

	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * 4).values.tolist()[0]

		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(24 * 5 * 2).values.tolist()

		back_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).values.tolist()

		if len(future_prices) < 12:
			continue

		deltas = []
		for index in range(1, 12):
			deltas.append(future_prices[index] - future_prices[0])

		for index in range(0, len(back_prices) - 1):
			deltas.append(back_prices[-1] - back_prices[index])

		calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= row["time"])]

		description_vector1_curr1 = [0] * len(description_map)
		description_vector2_curr1 = [0] * len(description_map)

		description_vector1_curr2 = [0] * len(description_map)
		description_vector2_curr2 = [0] * len(description_map)

		for index, history in calendar_history.iterrows():
			
			if history["currency"] == pair[0:3]:
				description_vector1_curr1[description_map[history["description"]]] = (history['actual'] - history['forecast'])
				description_vector2_curr1[description_map[history["description"]]] = (history['actual'] - history['previous'])
			else:
				description_vector1_curr2[description_map[history["description"]]] = (history['actual'] - history['forecast'])
				description_vector2_curr2[description_map[history["description"]]] = (history['actual'] - history['previous'])

		X_train.append(description_vector1_curr1 + description_vector2_curr1 + description_vector1_curr2 + description_vector2_curr2 + deltas)
		y_train.append(future_prices[-1] - future_prices[0])

	return X_train, y_train

def back_test(currency1, currency2, price_df, pair, year, test_calendar, description_map):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	print "no"

	equity = 0
	equity_curve = []
	orders = []
	returns = []

	end_of_year_time = test_calendar["time"].tail(1).values.tolist()[0]

	anchor_price = None
	prev_time = None

	last_train_time = 0
	for index, row in test_calendar.iterrows():

		if row["year"] != year:
			continue

		if row["time"] == prev_time:
			continue

		prev_time = row["time"]

		if abs(row["time"] - last_train_time) > 60 * 60 * 24 * 5 * 2:
			before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * 4 * 12).values.tolist()[0]
			start_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * 2).values.tolist()[0]
			calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= start_time)]

			print "in"
			X_train, y_train = create_training_set(currency1, currency2, price_df, pair, year, calendar_history, description_map)
			print "out"

			clf_mean = GradientBoostingRegressor(random_state=42)
			clf_mean.fit(X_train, y_train)
			last_train_time = row["time"]

		#print price_df[price_df['times'] >= row['time']]

		before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * 4).values.tolist()[0]
		future_times = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['times'].values.tolist()
		future_prices = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['prices'].values.tolist()
		future_mult_factor = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['mult_factor'].values.tolist()

		back_prices = (price_df[price_df['times'] <= row['time']])['prices'].tail(24).values.tolist()

		if len(future_prices) < 12:
			continue

		print "pass"

		deltas = []
		for index in range(1, 12):
			deltas.append(future_prices[index] - future_prices[0])

		for index in range(0, len(back_prices) - 1):
			deltas.append(back_prices[-1] - back_prices[index])

		future_prices = future_prices[12:]
		future_times = future_times[12:]
		future_mult_factors = future_mult_factor[12:]

		calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= row["time"])]

		description_vector1_curr1 = [0] * len(description_map)
		description_vector2_curr1 = [0] * len(description_map)

		description_vector1_curr2 = [0] * len(description_map)
		description_vector2_curr2 = [0] * len(description_map)

		for index, history in calendar_history.iterrows():
			
			if history["currency"] == pair[0:3]:
				description_vector1_curr1[description_map[history["description"]]] = (history['actual'] - history['forecast'])
				description_vector2_curr1[description_map[history["description"]]] = (history['actual'] - history['previous'])
			else:
				description_vector1_curr2[description_map[history["description"]]] = (history['actual'] - history['forecast'])
				description_vector2_curr2[description_map[history["description"]]] = (history['actual'] - history['previous'])

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]
		first_time = (price_df[price_df['times'] >= row['time']])['times'].head(1).values.tolist()[0]

		if anchor_price == None:
			anchor_price = first_price

		anchor_price = first_price
		anchor_time = first_time

		mean_prediction = clf_mean.predict([description_vector1_curr1 + description_vector2_curr1 + description_vector1_curr2 + description_vector2_curr2 + deltas])[0]

		max_total_pnl = equity
		max_delta = 0
		found_news_release = False
		time_step = 11

		curr_calendar_time = row["time"]
		for future_price, future_time, future_mult_factor in zip(future_prices, future_times, future_mult_factors):
			time_step += 1

			max_pnl = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					pnl = (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
				else:
					pnl = (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor

				max_pnl = max(max_pnl, pnl)

			total_pnl = 0
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					total_pnl += (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
				else:
					total_pnl += (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor

			max_total_pnl = max(max_total_pnl, equity + total_pnl)

			if abs(future_price - anchor_price) > max_delta * 2:
				delta = ((future_price - first_price) - mean_prediction) / 1

				new_order = Order()
				new_order.open_price = future_price
				new_order.open_time = future_time
				new_order.dir = delta < 0
				new_order.amount = 1.0 / (time_step * 0.1)

				orders.append(new_order)
				max_delta = abs(future_price - anchor_price)

			total_pnl = 0
			order_count = 0
			new_orders = []
			for order in orders:

				if len(orders) < 8 or order_count >= 1:
					new_orders.append(order)
				else:
					anchor_price = order.open_price
					max_delta = abs(future_price - anchor_price)
					if (future_price > order.open_price) == (order.dir):
						equity += (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
					else:
						equity += (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
					continue

				if (future_price > order.open_price) == (order.dir):
					total_pnl += (abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor
				else:
					total_pnl += (-abs(future_price - order.open_price) - commission) * order.amount * future_mult_factor

				order_count += 1

			orders = new_orders

			if (abs(future_price - anchor_price) < max_delta / (time_step * 0.1)):
				equity += total_pnl
				total_pnl = 0
				max_total_pnl = 0
				max_delta = 0
				anchor_price = first_price

				for order in orders:

					if (future_price > order.open_price) == (order.dir):
						returns.append(abs(future_price - order.open_price) * order.amount * future_mult_factor)
					else:
						returns.append(-abs(future_price - order.open_price) * order.amount * future_mult_factor)

				orders = []

				#break


		if found_news_release == False or True:
			anchor_price = None

			equity += total_pnl
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					returns.append(abs(future_price - order.open_price) * order.amount * future_mult_factor)
				else:
					returns.append(-abs(future_price - order.open_price) * order.amount * future_mult_factor)

			orders = []

		equity_curve.append(equity)
		print equity

	return equity_curve, returns


def create_data_set(year_range, currency_pair, select_year, description_map):

	X_train = []
	y_train = []

	X_test = []
	y_test = []

	for year in year_range:

		print "Year", year
		
		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 

			prices, times = load_time_series(test_pair, None)
			price_df = pd.DataFrame()
			price_df['prices'] = prices
			price_df['times'] = times

			test_calendar = get_calendar_df(test_pair, year)

			x, y = create_training_set(first_currency, second_currency, price_df, test_pair, year, test_calendar, description_map)


			if year != select_year:
				X_train += x
				y_train += y
			else:
				X_test += x
				y_test += y

	return X_train, y_train, X_test, y_test

'''
from sklearn import metrics
from sklearn.metrics import roc_auc_score, classification_report
import pickle


fpr = pickle.load(open("/tmp/fpr1", 'rb'))
tpr = pickle.load(open("/tmp/tpr1", 'rb'))
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

sys.exit(0)
'''

#["AUD_CAD", "GBP_CAD", "NZD_CAD", "AUD_NZD"]

total_pnl = 0
for currency_pair in ["NZD_CAD", "AUD_NZD"]:
	print currency_pair

	description_map = {}
	for year in range(2008, 2018):
		descriptions = get_calendar_df(currency_pair, year)["description"].values.tolist()
		for description in descriptions:
			if description not in description_map:
				description_map[description] = len(description_map)

	for year in range(2009, 2010):
		print year, currency_pair

		df1 = get_calendar_df(currency_pair, year-1)
		df2 = get_calendar_df(currency_pair, year)

		test_calendar = pd.concat([df1, df2])

		for test_pair in [currency_pair]:
			first_currency = test_pair[0:3] 
			second_currency = test_pair[4:7] 

			prices, times = load_time_series(test_pair, None)
			price_df = pd.DataFrame()
			price_df['prices'] = prices
			price_df['times'] = times
			price_df.reset_index(inplace=True)

			if test_pair[4:7] == "USD":
				price_df['mult_factor'] = 1.0
			elif test_pair[4:7] + "_USD" in currency_pairs:
				prices, times = load_time_series(test_pair[4:7] + "_USD", None)

				conv_df = pd.DataFrame()
				conv_df['mult_factor'] = prices
				conv_df['times'] = times
				price_df = price_df.set_index('times').join(conv_df.set_index('times'))
				price_df.reset_index(inplace=True)
			else:
				prices, times = load_time_series("USD_" + test_pair[4:7], None)

				conv_df = pd.DataFrame()
				conv_df['mult_factor'] = [1.0 / p for p in prices]
				conv_df['times'] = times
				price_df = price_df.set_index('times').join(conv_df.set_index('times'))
				price_df.reset_index(inplace=True)

			equity_curve, returns = back_test(first_currency, second_currency, price_df, test_pair, year, test_calendar, description_map)


			total_pnl += equity_curve[-1] 

			print "Sharpe", np.mean(returns) / np.std(returns)
			print "Total Profit", total_pnl

			import datetime
			import numpy as np
			from matplotlib.backends.backend_pdf import PdfPages
			import matplotlib.pyplot as plt


			with PdfPages('/Users/callummc/Desktop/equity_curve_' + str(year) + '_' + test_pair + '.pdf') as pdf:
				plt.figure(figsize=(6, 4))
				plt.plot(equity_curve)

				pdf.savefig()  # saves the current figure into a pdf page
				plt.close() 

