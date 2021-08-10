


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
	for pair in portfolio_wts['wt']:

		prices, times = get_time_series(pair, 4501)

		print len(prices), "***", times[-1]

		for i in range(4500):
			returns[i] += portfolio_wts['wt'][pair] * prices[i]


	labels = []
	for index in range(len(returns)):

	    if index < len(returns) - 48:
			labels.append(returns[index + 48] - returns[index])

	return [[v] for v in returns], labels

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger



import sqlite3

conn = sqlite3.connect('test.db')
print "Opened database successfully";


conn.execute('''
	CREATE TABLE IF NOT EXISTS CALENDAR_DAYS
         (DATE           CHAR(50));
         ''')

conn.execute('''
	CREATE TABLE IF NOT EXISTS CALENDAR_DATA
         (
         DATE           CHAR(50),
         TIMESTAMP      LONG,
         CURRENCY           CHAR(50),
         DESCRIPTION 
         ACTUAL           FLOAT,
         PREVIOUS           FLOAT,
         FORECAST           FLOAT,
         CONSTRAINT UC_Person UNIQUE (DESCRIPTION,CURRENCY,TIMESTAMP)
         );
         ''')

print "Table created successfully";

conn.close()

if get_mac() == 154505288144005:
	root_dir = "/tmp/"
else:
	root_dir = "/root/trading_data/"

trade_logger = setup_logger('first_logger', root_dir + "prediction_pairs_output.log")

