import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from bayes_opt import BayesianOptimization

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
import bisect

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


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

def get_deltas(currency_pair, times):

	before_prices, times, volumes = load_time_series(currency_pair, None)

	if currency_pair[4:7] == "JPY":
		pip_size = 0.01
	else:
		pip_size = 0.0001

	trend_map = {}
	for index in range(1, len(before_prices)):
		for time_index in [1, 2, 3, 4, 5, 10, 15, 20]:
			if time_index not in trend_map:
				trend_map[time_index] = []

			if index + time_index < len(before_prices):
				trend_map[time_index].append(abs((before_prices[-index] - before_prices[-time_index - 1 - index]) / pip_size))

	for time_index in [1, 2, 3, 4, 5, 10, 15, 20]:
		trend_map[time_index] = sorted(trend_map[time_index])


	time_df = pd.DataFrame()
	time_df["times"] = times
	time_df["prices"] = before_prices
	time_df.set_index("times", inplace=True)

	percentile_map = {}

	for time in times:
		percentile_map[time] = {}
		before_prices = (time_df[time_df.index <= time])["prices"].values.tolist()
		for time_index in [1, 2, 3, 4, 5, 10, 15, 20]:
			percentile_map[time][time_index] = {}
			delta = (before_prices[-1] - before_prices[-min(len(before_prices), time_index+1)]) / pip_size
			index = bisect.bisect(trend_map[time_index], abs(delta))
			if delta > 0:
				percentile_map[time][time_index]["percentile"] = int((float(index) / len(trend_map[time_index])) * 100)
			else:
				percentile_map[time][time_index]["percentile"] = -int((float(index) / len(trend_map[time_index])) * 100)

			percentile_map[time][time_index]["price"] = before_prices[-1]

	return percentile_map

def create_percentile_map():
	before_prices, times, volumes = load_time_series("AUD_CAD", None)

	times = [times[v] for v in range(0, len(times), 24)]

	percentile_map = {}
	for currency_pair in currency_pairs:

		print (currency_pair)
		percentile_map[currency_pair] = get_deltas(currency_pair, times)

		pickle.dump(percentile_map, open("pair_historic_percentile_map.pickle", "wb"))

#create_percentile_map()
#sys.exit(0)

class Order:

	def __init__(self):
		self.open_price = 0
		self.amount = 0
		self.dir = 0

def back_test():
	before_prices, times, volumes = load_time_series("AUD_CAD", None)
	times = [times[v] for v in range(0, len(times), 24)]
	percentile_map = pickle.load(open("pair_historic_percentile_map.pickle", "rb"))
	print ("loaded")
	orders = []
	net_profits = []
	indv_profits = []

	equity = 0
	existing_orders = set()
	for time in times:

		ranking = []
		pair_price = {}
		for currency_pair in currency_pairs[:3]:

			if time not in percentile_map[currency_pair]:
				continue

			for time_index in [1, 2, 3, 4, 5, 10, 15, 20]:
				ranking.append([currency_pair, percentile_map[currency_pair][time][time_index]["percentile"], percentile_map[currency_pair][time][time_index]["price"], time_index])
				pair_price[currency_pair] = percentile_map[currency_pair][time][time_index]["price"]

		ranking = sorted(ranking, key=lambda x: abs(x[1]), reverse=True)

		'''
		for index in range(int(len(ranking) * 0.4), int(len(ranking) * 0.5)):
			item = ranking[index]
			pair = item[0]
			percentile = item[1]
			price = item[2]

			order = Order()
			order.pair = pair
			order.open_price = price
			order.dir = (percentile < 0)
			order.amount = 1
			orders.append(order)

			pair_price[pair] = price
		'''
		
		
		for index in range(int(len(ranking) * 0.9), int(len(ranking) * 1.0)):
			item = ranking[index]
			pair = item[0]
			percentile = item[1]
			price = item[2]
			time_index = item[3]

			key = str(time_index) + pair
			if key not in existing_orders:
				existing_orders.add(key)

				order = Order()
				order.open_price = price
				order.pair = pair
				order.dir = (percentile > 0)
				order.amount = 1
				orders.append(order)

				pair_price[pair] = price
		


		indv_profits = indv_profits[-30:]
		net_profits = net_profits[-30:]

		profit = 0
		mean_profit = np.mean(indv_profits)
		std_profit = np.std(indv_profits)

		new_orders = []
		count = 0
		for order in orders:
			count += 1

			if order.pair[4:7] == "JPY":
				pip_size = 0.01
			else:
				pip_size = 0.0001

			if (order.dir) == (pair_price[order.pair] > order.open_price):
				pnl = abs(pair_price[order.pair] - order.open_price) / pip_size
			else:
				pnl = -abs(pair_price[order.pair] - order.open_price) / pip_size

			pnl -= 5

			indv_profits.append(pnl)
			new_orders.append(order)

			'''
			if count < 3 and len(orders) >= 20:
				equity += pnl
				continue
			'''

			if len(indv_profits) > 5:
				if abs(pnl - mean_profit) / std_profit > 2:
					equity += pnl
					continue

			new_orders.append(order)
			profit += pnl

		print (equity + profit)
		orders = new_orders

		net_profits.append(profit)

		if len(net_profits) > 5:
			mean_profit = np.mean(net_profits)
			std_profit = np.std(net_profits)

			if (profit - mean_profit) / std_profit > 2:
				equity += profit
				orders = []
				#net_profits = []
				#indv_profits = []
				existing_orders = set()

back_test()
sys.exit(0)

