


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


def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(5000) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=America%2FNew_York")

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
		prices.append(item['openMid'])
		index += 1

	return prices, times

'''currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]
'''

currency_pairs = [
     "GBP_AUD",
]

base_calendar = pd.DataFrame.from_records(download_calendar.download_calendar(31536000), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])

def find_stop_times(currency, price_df, pair):

	test_calendar = base_calendar[base_calendar['currency'] == currency]

	orders = []
	min_profit = 99999999

	equity = 0
	mag_factor = 1.0
	equity_curve = []

	stop_times = []
	for index, row in test_calendar.iterrows():

		#print price_df[price_df['times'] >= row['time']]

		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(30).values.tolist()[:12]

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(30).values.tolist()[:12]

		if abs(first_price[-1] - first_price[0]) > 0.001:
			stop_times.append(future_times[-1])

	return stop_times

def find_closest_distance(clusters, prediction):

	min_gap_price = 999999
	for price_index in range(len(clusters)):

		if price_index != prediction:
			min_gap_price = min(min_gap_price, abs(clusters[prediction][0] - clusters[price_index][0]))

	return min_gap_price

def find_max_distance_price(clusters, prediction):

	max_gap_price = 0
	max_price = 0
	for price_index in range(len(clusters)):

		if price_index != prediction:
			if abs(clusters[prediction][0] - clusters[price_index][0]) > max_gap_price:
				max_gap_price = abs(clusters[prediction][0] - clusters[price_index][0])
				max_price = clusters[price_index][0]

	return max_price

def back_test_pair(pair, prices, lows, highs, times, stop_times, index, orders, equity, equity_curve, margin_used, max_equity, start_equity, total_pnl, total_orders):

	price = prices[index]
	low_price = lows[index]
	high_price = highs[index]
	time = times[index]

	total_profit = 0
	buy_profit = 0
	sell_profit = 0

	buy_num = 0
	sell_num = 0
	buy_amount = 0
	sell_amount = 0
	max_buy_profit = -999999999
	max_sell_profit = -999999999

	min_buy_profit = 999999999
	min_sell_profit = 999999999

	last_buy_time = 0
	last_sell_time = 0
	local_margin_used = 0

	gap_mult = 1.0
	pair_mult = 1.0
	spread_mult = 0.01
	if pair[4:7] == "JPY":
		pair_mult /= 100
		gap_mult *= 100
		spread_mult = 1.0


	if time in stop_times:
		equity += total_profit
		total_profit = 0
		orders = []

	#print pair, total_profit, local_margin_used
	

	#print equity + total_profit, len(orders), buy_profit, sell_profit, max_buy_profit, max_sell_profit

	price_index = index + 1

	if buy_num == 0 or True:
		history_prices = prices[max(0, price_index-400):price_index]
		kmeans_buy = KMeans(n_clusters=min(30, len(history_prices)), init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit([[v] for v in history_prices])

	if sell_num == 0 or True:
		history_prices = prices[max(0, price_index-400):price_index]
		kmeans_sell = KMeans(n_clusters=min(30, len(history_prices)), init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit([[v] for v in history_prices])



	history_prices = prices[max(0, price_index-400):price_index]
	history_prices = [[v] for v in history_prices]

	buy_prediction = kmeans_buy.predict([history_prices[-1]])[0]
	buy_mean_center = kmeans_buy.cluster_centers_.tolist()[buy_prediction][0]

	sell_prediction = kmeans_sell.predict([history_prices[-1]])[0]
	sell_mean_center = kmeans_sell.cluster_centers_.tolist()[sell_prediction][0]

	min_sell = min([v[0] for v in kmeans_sell.cluster_centers_.tolist()])
	max_sell = max([v[0] for v in kmeans_sell.cluster_centers_.tolist()])

	min_buy = min([v[0] for v in kmeans_buy.cluster_centers_.tolist()])
	max_buy = max([v[0] for v in kmeans_buy.cluster_centers_.tolist()])

	stop_size = (max_buy - min_buy) * 1

	for order in orders:



		if order.dir == (prices[index-1] > order.open_price):
			pnl = abs(prices[index-1] - order.open_price) * order.amount
		else:
			pnl = -abs(prices[index-1] - order.open_price) * order.amount

		if order.dir == (price > prices[index-1]):
			pnl += abs(price - prices[index-1]) * order.amount
		else:
			pnl -= min(stop_size, abs(price - prices[index-1])) * order.amount 

		if pair[4:7] == "JPY":
			pnl /= 100

		if pnl > 0:
			pnl *= 0.9

		pnl *= order.equity_factor

		local_margin_used += order.amount

		total_profit += pnl
		if order.dir:
			buy_profit += pnl
			buy_num += 1
			buy_amount += order.amount
			max_buy_profit = max(max_buy_profit, pnl)
			min_buy_profit = min(min_buy_profit, pnl)
			last_buy_time = max(order.open_time, index)
		else:
			sell_profit += pnl
			sell_num += 1
			sell_amount += order.amount
			max_sell_profit = max(max_sell_profit, pnl)
			min_sell_profit = min(min_sell_profit, pnl)
			last_sell_time = max(order.open_time, index)




	new_orders = []
	first_order = False
	for order in orders:

		if order.dir == (prices[index-1] > order.open_price):
			pnl = abs(prices[index-1] - order.open_price) * order.amount
		else:
			pnl = -abs(prices[index-1] - order.open_price) * order.amount

		is_stop_activated = False
		if order.dir == False and (high_price - prices[index-1]) > stop_size:
			is_stop_activated = True

		if order.dir == True and (prices[index-1] - low_price) > stop_size:
			is_stop_activated = True

		if is_stop_activated:
			print "stoped out"
			pnl -= stop_size * order.amount 
			equity += pnl
			#total_profit -= pnl
			max_equity = min(max_equity, equity)
			continue

		if order.dir == (price > prices[index-1]):
			pnl += abs(price - prices[index-1]) * order.amount
		else:
			pnl -= min(stop_size, abs(price - prices[index-1])) * order.amount 


		if pair[4:7] == "JPY":
			pnl /= 100

		if pnl > 0:
			pnl *= 0.9
	
		
		if (start_equity + total_pnl > max_equity):

			equity += pnl
			#total_profit -= pnl
			prev_buy_cluster = 0

			continue
		
		
		if index > 0 and abs(price - prices[index-1]) > stop_size:
			equity += pnl
			#total_profit -= pnl
			max_equity = min(max_equity, equity)

			if pnl < 0:
				print "Clsoe"
			

			continue

		'''
		if order.dir == (buy_num > sell_num) and start_equity + total_pnl > max_equity and first_order == False and total_orders >= 200:
			equity += pnl
			#total_profit -= pnl
			max_equity = min(max_equity, equity)
			first_order = True
			print "max_exposure"
			continue
		'''
		
		'''
		if order.dir == (buy_num > sell_num) and max(buy_num, sell_num) >= 15 and first_order == False:
			equity += pnl
			#total_profit -= pnl
			max_equity = min(max_equity, equity)
			first_order = True
			print "max_exposure"
			continue
		'''
		
		if (margin_used > (equity + min(0, total_profit)) * 50) or min(buy_num, sell_num) >= 5:
			equity += pnl
			#total_profit -= pnl
			max_equity = min(max_equity, equity)
			first_order = True

			continue

		'''
		if (total_profit / equity < -0.02):
			equity += pnl
			total_profit -= pnl
			max_equity = min(max_equity, equity)
			first_order = True

			continue
		'''

		new_orders.append(order)
		

	orders = new_orders

	#print pair, min(buy_num, sell_num), max(buy_num, sell_num)

	import random

	if True:#local_margin_used <= (abs(total_profit) + 1) * 1000  * (min(buy_num, sell_num) + 1):

		if (max_buy_profit < 0) or buy_num == 0 or (index - last_buy_time) > 24 / (sell_num + 1):

			if True:#abs(price - sell_mean_center) > spread_mult * 0.001 * (min(buy_num, sell_num) + 1):#find_closest_distance(kmeans_buy.cluster_centers_.tolist(), buy_prediction) < 0.001 * pair_mult:
				new_order = Order()
				new_order.open_price = price
				new_order.dir = True
				new_order.amount = round(spread_mult * spread_mult * (equity / 10000) *  (max(buy_num, sell_num) + 1) / (max(spread_mult * 0.001, 1) * max(spread_mult * 0.003, max_buy - min_buy)))
				new_order.open_time = index
				new_order.equity_factor = 1#equity / 5000
				orders.append(new_order)

				prev_buy_cluster = buy_mean_center
				prev_sell_cluster = sell_mean_center

		if (max_sell_profit < 0) or sell_num == 0 or (index - last_sell_time) > 24 / (buy_num + 1):

			if True:#abs(price - sell_mean_center) > spread_mult * 0.001 * (min(buy_num, sell_num) + 1):#find_closest_distance(kmeans_sell.cluster_centers_.tolist(), sell_prediction) < 0.001 * pair_mult:
				new_order = Order()
				new_order.open_price = price
				new_order.dir = False
				new_order.amount = round(spread_mult * spread_mult * (equity / 10000) * (max(buy_num, sell_num) + 1) / (max(spread_mult * 0.001, 1) * max(spread_mult * 0.0003, max_sell - min_sell)))
				new_order.open_time = index
				new_order.equity_factor = 1#equity / 5000
				orders.append(new_order)

				prev_sell_cluster = sell_mean_center
				prev_buy_cluster = buy_mean_center

	equity_curve.append(equity + total_profit)

	return orders, equity, equity_curve, max_equity


def load_time_series(symbol):

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


	rates = []
	prices = []
	labels = []
	lows = []
	highs = []
	price_range = []

	content = content[1:]

	for index in range(len(content)):

		toks = content[index].split(',')

		high = float(toks[2])
		low = float(toks[3])
		o_price = float(toks[1])
		c_price = float(toks[4])

		rates.append([high - low, c_price - o_price])
		prices.append(c_price)
		price_range.append(c_price - o_price)

		lows.append(low)
		highs.append(high)

		if index < len(content) - 48:

			toks = content[index + 48].split(',')

			labels.append(float(toks[4]) - c_price)

	return prices, lows, highs


total_return = 0

returns = []
equity_curves = []
titles = []

index = 0
equity_curve = []
orders_pair = {}
equity = 100000
max_equity = equity


all_pairs = [
    "CHF_JPY", "GBP_JPY",  
    "AUD_JPY",
    "EUR_JPY", "NZD_JPY", 
    "CAD_JPY", "USD_JPY"
]




all_pairs = [
    "AUD_CAD", "EUR_NZD", 
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "GBP_CAD", 
    "GBP_CHF", "NZD_USD", 
]


all_pairs = [
    "CHF_JPY", "GBP_JPY",  
    "AUD_JPY",
    "EUR_JPY", "NZD_JPY", 
    "CAD_JPY", "USD_JPY"
]


all_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

all_pairs = [
    "AUD_NZD", "AUD_CAD", "EUR_AUD", "GBP_AUD", "AUD_CHF", "AUD_JPY",
]

all_pairs = [
    "AUD_NZD", "NZD_CAD", "EUR_NZD", "GBP_NZD", "NZD_CHF", "NZD_JPY",
]


all_pairs = [
    "CHF_JPY", "GBP_JPY",  
    "AUD_JPY",
    "EUR_JPY", "NZD_JPY", 
    "CAD_JPY", "USD_JPY"
]


all_pairs = [
    "GBP_AUD", "GBP_NZD", "EUR_GBP", "GBP_CAD", "GBP_CHF", "GBP_JPY",
]


all_pairs = [
    "EUR_AUD", "EUR_NZD", "EUR_GBP", "EUR_CAD", "EUR_CHF", "EUR_JPY",
]




all_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", 
    "AUD_JPY", "EUR_CAD", 
    "AUD_NZD", "EUR_CHF", "NZD_CAD", 
    "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", 
]

all_pairs = [
    "AUD_CAD", "EUR_NZD", "GBP_JPY",  
     "EUR_AUD", "GBP_NZD", 
    "AUD_JPY", "EUR_CAD", 
    "AUD_NZD",  "NZD_CAD", 
    "EUR_GBP", "GBP_AUD", 
     "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", 
]


all_pairs = [
    "AUD_NZD", "AUD_CAD", "EUR_AUD", "GBP_AUD", "AUD_CHF", "AUD_JPY",
]


total_size = 0
test_prices = {}
test_lows = {}
test_highs = {}
for test_pair in all_pairs:
	prices, lows, highs = load_time_series(test_pair)
	total_size = len(prices)

	test_prices[test_pair] = prices[-40000:-30000]
	test_lows[test_pair] = lows[-40000:-30000]
	test_highs[test_pair] = highs[-40000:-30000]

print total_size
offset_index = 0
returns = []
while offset_index < 30000:

	margin_used = 0
	total_pnl = 0
	total_orders = 0
	for test_pair in all_pairs:

		if test_pair not in orders_pair:
			orders_pair[test_pair] = []

		orders = orders_pair[test_pair]
		prices = test_prices[test_pair]

		price = prices[offset_index]

		gap_mult = 1.0
		pair_mult = 1.0
		spread_mult = 0.01
		if test_pair[4:7] == "JPY":
			pair_mult /= 100
			gap_mult *= 100
			spread_mult = 1.0


		for order in orders:
			total_orders += 1
			margin_used += order.amount * order.equity_factor

			if order.dir == (prices[offset_index-1] > order.open_price):
				pnl = abs(prices[offset_index-1] - order.open_price) * order.amount
			else:
				pnl = -abs(prices[offset_index-1] - order.open_price) * order.amount

			if order.dir == (price > prices[offset_index-1]):
				pnl += abs(price - prices[offset_index-1]) * order.amount
			else:
				pnl -= min(0.02 * gap_mult, abs(price - prices[offset_index-1])) * order.amount 

			if test_pair[4:7] == "JPY":
				pnl /= 100

			if pnl > 0:
				pnl *= 0.9

			total_pnl += pnl

	start_equity = equity

	for test_pair in all_pairs:
		first_currency = test_pair[0:3] 
		second_currency = test_pair[4:7] 

		if test_pair not in orders_pair:
			orders_pair[test_pair] = []

		prices = test_prices[test_pair]
		lows = test_lows[test_pair]
		highs = test_highs[test_pair]


		orders_pair[test_pair], equity, equity_curve, max_equity = back_test_pair(test_pair, prices, lows, highs, [0] * len(prices), [], offset_index,
			orders_pair[test_pair], equity, equity_curve, margin_used, max_equity, start_equity, total_pnl, total_orders)

	total_orders = 0
	for test_pair in all_pairs:
		total_orders += len(orders_pair[test_pair])

	print equity_curve[-1], (offset_index / 24), total_orders, total_pnl

	returns.append(equity - start_equity)
	if (equity + total_pnl > max_equity):
		max_equity = equity + total_pnl

	offset_index += 1

equity_curves.append(equity_curve)
print "Sharpe", (np.mean(returns) / np.std(returns)) * math.sqrt(4 * 250)
print "Total Return", total_return
	
print all_pairs


import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


with PdfPages('/Users/callummc/Desktop/equity_curve.pdf') as pdf:

	for equity_curve, title in zip(equity_curves, titles):
		plt.figure(figsize=(6, 4))
		plt.plot(equity_curve)
		plt.title(title)

		pdf.savefig()  # saves the current figure into a pdf page
		plt.close() 



