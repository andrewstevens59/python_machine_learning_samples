



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

from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from evaluate_model import evaluate

import delta_process as delta_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import train_and_back_test_all as back_test_all
#import matplotlib.pyplot as plt

import plot_equity as portfolio
from uuid import getnode as get_mac
import socket

import back_test_strategies as back_test
import random

from maximize_sharpe import *
import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import grid_delta as grid_delta
from uuid import getnode as get_mac
import logging
import socket


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


trade_bounds = [[2, 1000]]
avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))

existing_fifo_order = set()

model_predictions = pickle.load(open("/tmp/model_predictions", 'rb'))
model_prices = pickle.load(open("/tmp/model_prices", 'rb'))

model_price_change = pickle.load(open("/tmp/model_price_change", 'rb'))


global_returns = pickle.load(open("/tmp/global_returns", 'rb'))
global_currency_pairs = pickle.load(open("/tmp/global_currency_pairs", 'rb'))
times  = pickle.load(open("/tmp/times", 'rb'))

sharpes = []
est_sharpes = []
returns = []
time_exposure = {}
time_exp_insts = {}

from datetime import datetime as dt
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
	price_range = []
	volumes = []

	for index in range(len(j) - 1):
		item = j[index]

		s = item['time'].replace(':', "-")
		s = s[0 : s.index('.')]

		date_obj = dt.strptime(s, "%Y-%m-%dT%H-%M-%S")
		item['time'] = dt.strftime(date_obj, "%Y.%m.%d %H:%M:%S")

		times.append(item['time'])
		prices.append([item['closeMid']])
		volumes.append([item['volume']])

		if index < len(j) - 48:
			rates.append(j[index + 47]['closeMid'] - j[index]['closeMid'])
			labels.append(j[index + 47]['closeMid'] - j[index]['closeMid'])

	return rates, prices, labels, price_range, times, volumes

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
	price_range = []
	times = []

	content = content[1:]

	for index in range(len(content)):

		toks = content[index].split(',')

		high = float(toks[2])
		low = float(toks[3])
		o_price = float(toks[1])
		c_price = float(toks[4])

		rates.append([high - low, c_price - o_price])
		prices.append([c_price])
		price_range.append(c_price - o_price)
		times.append(toks[0])

		if index < len(content) - 48:

			toks = content[index + 48].split(',')

			labels.append(float(toks[4]) - c_price)

	return times, labels


year = 2013




if os.path.isfile("/tmp/all_times" + str(year)) == False:
	offset = 0
	models = []
	all_times = set()
	new_model_predictions = []
	for pair in currency_pairs:

		times, labels = load_time_series(pair)

		#rates, prices, labels, price_range, times, volumes = get_time_series(pair, 0)

		for model_index in range(3):
			start = 0
			end = 700
			time_set = []

			model_output = {}
			while end < len(labels):

				if times[end - 1] > str(year) and times[end - 1] <= str(year + 1) and len(time_set) < len(model_predictions[offset]):

					for i in range(start, end):
						model_output[times[i]] = {
							'prediction' : model_predictions[offset][len(time_set)],
							'price' : model_prices[offset][len(time_set)][0],
						}

					all_times.add(times[end - 1])

				time_set.append(times[end - 1])

				new_model_predictions.append(model_predictions[offset])
				start += 12
				end += 12

			models.append(model_output)
			offset += 1
			#offset += 1

	all_times = list(all_times)
	all_times = sorted(all_times)

	print offset, len(model_predictions)

	pickle.dump(all_times, open("/tmp/all_times" + str(year), 'wb'))
	pickle.dump(models, open("/tmp/models" + str(year), 'wb'))

all_times = pickle.load(open("/tmp/all_times" + str(year), 'rb'))
models = pickle.load(open("/tmp/models" + str(year), 'rb'))


global_avg = []
global_time_spans = []
global_returns = []
for currency in ["EUR", "USD", "CAD", "JPY", "NZD", "CHF", "GBP", "AUD"]:
	avg_close_days = []
	equity_curve = [1.0]
	start_equity = 100000

	trade_swap = 1
	for i in range(1):

		strategies = []
		reduce_order_map = {}
		prev_dir_map = {}
		model_orders = {}

		model_weights = {}
		for pair in currency_pairs:
			copy_returns = []
			for model_index in range(3):

				#copy_returns.append(new_model_predictions[len(strategies)])

				for trade_side in [None]:

					for bound in trade_bounds:

						model_orders[len(strategies)] = back_test.Model()
						prev_dir_map[len(strategies)] = None
						reduce_order_map[len(strategies)] = 1

						strategies.append(back_test.Tuple(bound, pair, trade_side, model_index))

			'''
			copy_returns = map(list, zip(*copy_returns))
			portfolio_wts = max_sharpe_ratio(copy_returns, optimization_bounds = (0.0, 1.0))
			model_weights[pair] = portfolio_wts
			'''
			
		start_equity, return_val, is_finish, time_span = back_test.back_test_strategies(trade_swap, start_equity, currency, i, time_exposure, time_exp_insts, model_orders, strategies, models, all_times, prev_dir_map, reduce_order_map, avg_spreads, avg_prices, model_weights)
		
	global_returns.append((start_equity / 100000) - 1)
	print currency, global_returns[-1]
		
print "Global", np.sum(global_returns)

	
	
