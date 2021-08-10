


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
import datetime as dt
import logging
import socket


def load_time_series(symbol, time):

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

		if index < len(content) - 48:

			toks = content[index + 48].split(',')

			labels.append(float(toks[4]) - c_price)

	return rates, prices, labels, price_range


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


def get_time_series(symbol, time):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	prices = []
	times = []

	for index in range(len(j) - 1):
	    item = j[index]

	    times.append(item['time'])
	    prices.append(item['closeMid'])

	return prices, times

def calculate_stat_arb_series(portfolio_wts):

	returns = [0] * 4500
	pair_prices = {}
	for pair in portfolio_wts['wt']:

		prices, times = get_time_series(pair, 4501)

		pair_prices[pair] = []

		print len(prices), "***", times[-1]

		for i in range(4500):
			returns[i] += portfolio_wts['wt'][pair] * prices[i]
			pair_prices[pair].append(prices[i])


	labels = {}
	series = {}
	for pair in portfolio_wts['wt']:

		labels[pair] = []
		series[pair] = []
		for index in range(len(returns)):

			series[pair].append([(pair_prices[pair][index] * portfolio_wts['wt'][pair]) - returns[index]])

			if index < len(pair_prices[pair]) - 48:
				labels[pair].append(pair_prices[pair][index + 48] - pair_prices[pair][index])

	return series, labels

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if get_mac() == 154505288144005:
	root_dir = "/tmp/"
else:
	root_dir = "/root/trading_data/"

trade_logger = setup_logger('first_logger', root_dir + "prediction_basket_output.log")

x1 = breakout_process.Breakout() # bad
x2 = volatility_process.VolatilityProcess()# bad
x3 = delta_process.DeltaProcess()#-2.17
x4 = jump_process.JumpProcess()#okay
x5 = create_regimes.CreateRegimes() # good
x6 = gradient.Gradient()# good
x7 = gradient.Gradient()#
x8 = barrier.Barrier()# excellent
x9 = barrier.Barrier()#
x10 = grid_delta.GridDelta()# okay
x11 = markov_process.MarkovProcess()

x_sets = [x5, x2, x4, x11, x10, x6]
lags = [0, 0, 0, 0, 0, 96]

portfolio_wts = pickle.load(open(root_dir + "portfolio_wts_basket", 'rb'))

prices, y = calculate_stat_arb_series(portfolio_wts)

white_list = pickle.load(open(root_dir + "basket_model_whitelist", 'rb'))
model_prediction_set = []

prediction_key_map = {}

for item in white_list:

	model_key = item['model_key']
	prediction_key = item['prediction_key']
	is_use_residual = item['is_use_residual']
	pair = item['currency_pairs'][0]
	item['weight'] = 1.0

	if prediction_key not in prediction_key_map or True:
		history_prices = pickle.load(open(root_dir + "stat_arb_series_" + prediction_key, 'rb'))

		history_prices = history_prices[prediction_key]
		history_prices = [[v] for v in history_prices]

	trade_logger.info('model_key: ' + model_key) 

	for model, lag in zip(x_sets, lags):

		model_key_base = prediction_key + "_" + str(model.__class__.__name__) + "_" + str(lag)

		if model_key != model_key_base:
			continue

		if lag == 0:
			prediction = model.make_prediction(model_key, None, prices[pair], y[pair], None, history_prices, is_use_residual)
		else:
			prediction = model.make_prediction(model_key, None, prices[pair], y[pair], None, lag, history_prices, is_use_residual)


		print model_key_base, "Prediction", prediction, item['trade_dir'], item['entry_bias'], "Weight", item['weight']

		trade_logger.info('Model: ' + model_key_base + ", Prediction: " + str(prediction) + ", Weight: " + str(item['weight'])) 

		model_prediction_set.append({
			'model_key' : model_key, 
			'prediction_key' : prediction_key,
			'prediction' : prediction, 
			'currency_pairs' : item['currency_pairs'],
			'wts' : item['wts'],
			'trade_dir' : item['trade_dir'],
			'weight' : item['weight'],
			'entry_bias' : item['entry_bias'],
			'is_use_residual' : item['is_use_residual'],
			'martingale_type' : item['martingale_type']
			})

	pickle.dump(model_prediction_set, open("/tmp/model_basket_predictions", 'wb'))

trade_logger.info('Done, Total: ' + str(len(white_list)))

