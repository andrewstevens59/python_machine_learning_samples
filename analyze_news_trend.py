


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
import re

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


def back_test_currency(currency1, currency2, price_df, pair, year):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission = avg_spreads[pair] * pip_size

	test_calendar = get_calendar_df(pair, year)

	count_time_stamp = {}
	for index, row in test_calendar.iterrows():
		prev_time = row['time']

		if prev_time not in count_time_stamp:
			count_time_stamp[prev_time] = 0

		count_time_stamp[prev_time] += 1

	orders = []
	min_profit = 99999999

	equity = 0
	mag_factor = 1.0
	equity_curve = []
	prev_time = None
	for index, row in test_calendar.iterrows():

		
		if row['time'] == prev_time:
			continue
		

		#print price_df[price_df['times'] >= row['time']]

		prev_time = row['time']
		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(38).values.tolist()[12:36]

		if len(future_times) < 12:
			continue

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]
	
		reset_treshold = 0

		max_equity = equity
		for future_time, time_index in zip(future_times, range(len(future_times))):

			prices = (price_df[price_df['times'] >= row['time']])['prices'].head(38).values.tolist()
			times = (price_df[price_df['times'] >= row['time']])['times'].head(38).values.tolist()

			price_deltas = []
			time_offset = 1
			curr_price = None
			for price_v, time_v in zip(prices, times):

				if time_v <= future_time:
					price_deltas.append((price_v - first_price) / 1)
					curr_price = price_v

				time_offset += 1

			std = np.std(price_deltas)
			if pair[4:7] != "JPY":
				std *= 100

			sharpe = (price_deltas[-1] - np.mean(price_deltas)) / np.std(price_deltas)

			if len(orders) == 0:
				min_profit = 99999999

			buy_amount = 0
			sell_amount = 0
			for order in orders:
				if order.dir:
					buy_amount += order.amount
				else:
					sell_amount += order.amount


			total_profit = 0
			order_count = 0
			for order in orders:
				if order.open_time > row['time']:
					continue

				if order.dir == (first_price > order.open_price):
					pnl = (abs(first_price - order.open_price) - commission) * order.amount
				else:
					pnl = -(abs(first_price - order.open_price) + commission) * order.amount

				if pair[4:7] != "JPY":
					pnl *= 100

				total_profit += pnl

			if abs(sharpe) > 2:
				new_order = Order()
				new_order.open_price = curr_price
				new_order.dir = sharpe < 0
				new_order.amount = (1 + abs(total_profit))
				new_order.open_time = future_time
				orders.append(new_order)
				reset_treshold = abs(sharpe) + 0.1
				#mag_factor *= 2
				mag_factor += 1
				break

		total_profit = 0
		order_count = 0
		for order in orders:
			if order.open_time > row['time']:
				continue

			if order.dir == (first_price > order.open_price):
				pnl = (abs(first_price - order.open_price) - commission) * order.amount
			else:
				pnl = -(abs(first_price - order.open_price) + commission) * order.amount

			if pair[4:7] != "JPY":
				pnl *= 100

			total_profit += pnl
			order_count += 1

		equity_curve.append(equity + total_profit)

		
		if total_profit > 0 or order_count >= 4:
			equity += total_profit
			mag_factor = 1.0
			orders = []
		

	return equity + total_profit, equity_curve


for year in range(2009, 2018):

	print "Year", year 
	total_return = 0

	returns = []
	equity_curves = []
	titles = []
	equity_curve_map = {}
	for test_pair in ["AUD_CAD", "AUD_NZD", "NZD_CAD"]:
		first_currency = test_pair[0:3] 
		second_currency = test_pair[4:7] 

		print test_pair

		prices, times = load_time_series(test_pair, year)

		
		price_df = pd.DataFrame()
		price_df['prices'] = prices
		price_df['times'] = times

		r1, equity_curve = back_test_currency(first_currency, second_currency, price_df, test_pair, year)

		equity_curve_map[test_pair] = equity_curve

		equity_curves.append(equity_curve)
		titles.append(test_pair + "_linear")


		returns.append(r1)

		total_return += r1

		print total_return

	pickle.dump(equity_curve_map, open("/Users/callummc/equity_curve_set_" + str(year), 'wb'))

	print "Sharpe", np.mean(returns) / np.std(returns)
	print "Total Return", total_return
		


	import datetime
	import numpy as np
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt


	with PdfPages('/Users/callummc/Desktop/equity_curve_' + str(year) + '.pdf') as pdf:

		for equity_curve, title in zip(equity_curves, titles):
			plt.figure(figsize=(6, 4))
			plt.plot(equity_curve)
			plt.title(title)

			pdf.savefig()  # saves the current figure into a pdf page
			plt.close() 



