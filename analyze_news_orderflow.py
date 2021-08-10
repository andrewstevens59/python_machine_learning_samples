


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
		prices.append(item['closeMid'])
		index += 1

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

base_calendar = pd.DataFrame.from_records(download_calendar.download_calendar(31536000), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])

def back_test_currency(currency, price_df, pair):

	test_calendar = base_calendar[base_calendar['currency'] == currency]

	orders = []
	min_profit = 99999999

	equity = 0
	mag_factor = 1.0
	equity_curve = []
	for index, row in test_calendar.iterrows():

		#print price_df[price_df['times'] >= row['time']]

		future_times = (price_df[price_df['times'] >= row['time']])['times'].head(30).values.tolist()[12:36]

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]

		reset_treshold = 0

		max_equity = equity
		for calendar_time, time_index in zip(future_times, range(len(future_times))):

			prices = (price_df[price_df['times'] >= calendar_time])['prices'].head(30).values.tolist()[:12]
			times = (price_df[price_df['times'] >= calendar_time])['times'].head(30).values.tolist()[:12]

			price_deltas = []
			time_offset = 1
			for price_v in prices:
				price_deltas.append((price_v - first_price) / 1)
				time_offset += 1

			std = np.std(price_deltas)
			if pair[4:7] != "JPY":
				std *= 100

			sharpe = (price_deltas[-1] - np.mean(price_deltas)) / np.std(price_deltas)

			curr_price = prices[-1]
			curr_time = times[-1]

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
			for order in orders:

				if order.dir == (curr_price > order.open_price):
					pnl = abs(curr_price - order.open_price) * order.amount
				else:
					pnl = -abs(curr_price - order.open_price) * order.amount

				if pair[4:7] != "JPY":
					pnl *= 100

				if (order.dir != (sharpe < 0)):
					total_profit += pnl

			if abs(sharpe) > 2:
				new_order = Order()
				new_order.open_price = curr_price
				new_order.dir = sharpe > 0
				new_order.amount = 1 
				new_order.open_time = curr_time
				orders.append(new_order)
				reset_treshold = abs(sharpe) + 0.1
				#mag_factor *= 2
				mag_factor += 1
				break
		

			total_profit = 0
			new_orders = []
			for order in orders:

				if order.dir == (curr_price > order.open_price):
					pnl = abs(curr_price - order.open_price) * order.amount
				else:
					pnl = -abs(curr_price - order.open_price) * order.amount

				if pair[4:7] != "JPY":
					pnl *= 100

				total_profit += pnl

				new_orders.append(order)


			orders = new_orders

			min_profit = min(min_profit, total_profit)

		total_profit = 0
		for order in orders:

			if order.dir == (curr_price > order.open_price):
				pnl = abs(first_price - order.open_price) * order.amount
			else:
				pnl = -abs(first_price - order.open_price) * order.amount

			if pair[4:7] != "JPY":
				pnl *= 100

			total_profit += pnl

		equity_curve.append(equity + total_profit)

		if total_profit > 0 or mag_factor >= 8:
			equity += total_profit
			mag_factor = 1.0
			orders = []

	return equity + total_profit, equity_curve


total_return = 0

returns = []
equity_curves = []
titles = []
for test_pair in currency_pairs:
	first_currency = test_pair[0:3] 
	second_currency = test_pair[4:7] 


	print first_currency
	prices, times = get_time_series(test_pair, 5000) 

	
	price_df = pd.DataFrame()
	price_df['prices'] = prices
	price_df['times'] = times

	r1, equity_curve = back_test_currency(first_currency, price_df, test_pair)

	equity_curves.append(equity_curve)
	titles.append(test_pair + "_" + first_currency + "_linear")


	print second_currency
	r2, equity_curve = back_test_currency(second_currency, price_df, test_pair)

	equity_curves.append(equity_curve)
	titles.append(test_pair + "_" + second_currency + "_linear")

	returns.append(r1 + r2)

	total_return += r1 + r2

	print total_return

print "Sharpe", np.mean(returns) / np.std(returns)
print "Total Return", total_return
	


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



