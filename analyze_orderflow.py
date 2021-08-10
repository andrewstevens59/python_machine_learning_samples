


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


def get_order_book(symbol, time, curr_price):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(pycurl.ENCODING, 'gzip') 
	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v3/instruments/" + symbol + "/orderBook?time=" + time)

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8', 'Content-Type: application/json'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['orderBook']['buckets']

	net_sum = 0
	for item in j:

		if abs(float(item['price']) - curr_price) >  0:
			net_sum += (float(item['longCountPercent']) - float(item['shortCountPercent'])) / abs(float(item['price']) - curr_price)

	return net_sum

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

		times.append(item['time'])
		prices.append(item['closeMid'])


		net_sum = get_order_book(symbol, item['time'], prices[-1])

		if len(times) >= 4:

			balances = [0] * 4

			for offset in range(1, 5):

				if times[-offset] not in balance_map:
					balances[offset-1] = get_order_book(symbol, times[-offset], prices[-offset])
					balance_map[times[-offset]] = balances[offset-1]
				else:
					balances[offset-1] = balance_map[times[-offset]]
					print "ues"

			features = [balances[0]]
			features += [balances[0] - balances[1]]
			features += [balances[0] - balances[2]]
			features += [balances[0] - balances[3]]

			if index < len(j) - 48:
				X.append(features)
				y.append(j[index + 47]['closeMid'] - j[index]['closeMid'])

			print len(X), len(y), "***", len(features)

		index += 12

	return X, y

currency_pairs = [
    "GBP_JPY",  
    "EUR_AUD", "USD_CAD", 
    "AUD_JPY", "GBP_USD", "USD_CHF", 
    "EUR_CHF", "EUR_USD", 
    "AUD_USD", "EUR_GBP",
    "EUR_JPY",
    "GBP_CHF", "NZD_USD", "USD_JPY"
]


mean_returns = []

returns = None
for test_pair in currency_pairs:

	X_train = []
	X_test = []

	y_train = []
	y_test = []

	for pair in currency_pairs:

		print pair


		if os.path.isfile("/tmp/X_" + pair) == False:
			X, y = get_time_series(pair, 5000)
			pickle.dump(X, open("/tmp/X_" + pair, 'wb'))
			pickle.dump(y, open("/tmp/y_" + pair, 'wb'))
		else:
			X = pickle.load(open("/tmp/X_" + pair, 'rb'))
			y = pickle.load(open("/tmp/y_" + pair, 'rb'))

		mean_y = np.mean([abs(y1) for y1 in y])
		y = [y1 / mean_y for y1 in y]

		if returns == None:
			returns = [0] * len(y)

		if pair == test_pair:
			X_test += X
			y_test += y
	
		else:
			X_train += X
			y_train += y


	print "testing", test_pair
	clf = GradientBoostingRegressor()
	clf.fit(X_train, y_train)


	predictions = clf.predict(X_test)

	mean = np.mean(predictions)
	std = np.std(predictions)

	pickle.dump({"mean" : mean, "std" : std}, open("/tmp/orderflow_model_stats", 'wb'))

	predictions = [(p - mean) / std for p in predictions]

	equity = 0
	index = 0
	for y, p in zip(y_test, predictions):

		if abs(p) > 0:

			if (p > 0) == (y > 0):
				equity += abs(y) * abs(p)
				returns[index] += abs(y) * abs(p) 
			else:
				equity -= abs(y) * abs(p)
				returns[index] -= abs(y) * abs(p)

		index += 1

	mean_returns.append(equity)

equity_curve = [0]
for index in range(len(returns)):
	equity_curve.append(returns[index] + equity_curve[-1])

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


with PdfPages('/Users/callummc/Desktop/equity_curve.pdf') as pdf:

	plt.figure(figsize=(6, 4))
	plt.plot(equity_curve)
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close() 

sys.exit(0)

print "Mean", np.mean(mean_returns)



