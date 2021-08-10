


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


def get_calendar_df(pair, year): 

	currencies = [pair[0:3], pair[4:7]]

	with open("/tmp/trading_data_sample/calendar_" + str(year) + ".txt") as f:
		content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	lines = [x.strip() for x in content] 

	from_zone = tz.gettz('America/New_York')
	to_zone = tz.gettz('UTC')

	contents = []

	for line in lines:
		line = line[len("2018-12-23 22:44:55 "):]
		toks = line.split(",")

		if toks[2] in currencies:

			est = datetime.datetime.strptime(toks[0] + " " + toks[1], "%b%d.%Y %H:%M%p")
			est = est.replace(tzinfo=from_zone)
			utc = est.astimezone(to_zone)

			time = calendar.timegm(utc.timetuple())

			contents.append([toks[2], time])

	return pd.DataFrame(contents, columns=["currency", "time"])



def load_time_series(symbol, year):

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir('/tmp/trading_data_sample/') if isfile(join('/tmp/trading_data_sample/', f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Ask' not in file:
			break

	with open('/tmp/trading_data_sample/' + file) as f:
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


def create_training_set(currency1, currency2, price_df, pair, year, test_calendar):


	return X_train, y_train_mean, model

def back_test(currency1, currency2, price_df, pair, year, test_calendar, model):



	return equity_curve




def create_data_set(year_range, currency_pair, select_year):




total_pnl = 0
for currency_pair in currency_pairs:
	print currency_pair

	year = 2017
	X_train, y_train_mean, y_train_std, kmeans, X_rev, y_rev = create_data_set(range(2009, 2018), currency_pair, year)

	print "revert", len(X_rev), len(y_rev)


	for test_pair in [currency_pair]:
		first_currency = test_pair[0:3] 
		second_currency = test_pair[4:7] 

		#X_train = pickle.load(open("/tmp/X_train", 'rb'))
		#y_train = pickle.load(open("/tmp/y_train", 'rb'))

		from sklearn.ensemble import GradientBoostingRegressor
		from sklearn.ensemble import RandomForestRegressor
		from sklearn.model_selection import RandomizedSearchCV
		import numpy as np
 
 		'''
		clf = GradientBoostingRegressor(verbose=10, random_state=42)

		loss = ['ls', 'lad', 'huber']
		n_estimators = [100, 500, 900, 1100, 1500]
		max_depth = [2, 3, 5, 10, 15]
		min_samples_leaf = [1, 2, 4, 6, 8] 
		min_samples_split = [2, 4, 6, 10]
		max_features = ['auto', 'sqrt', 'log2', None]

		# Define the grid of hyperparameters to search
		hyperparameter_grid = {'loss': loss,
		    'n_estimators': n_estimators,
		    'max_depth': max_depth,
		    'min_samples_leaf': min_samples_leaf,
		    'min_samples_split': min_samples_split,
		    'max_features': max_features}

		clf = RandomizedSearchCV(clf, hyperparameter_grid, cv=5, scoring = 'neg_mean_absolute_error')

		clf.fit(X_train, y_train_mean)
		pickle.dump(clf, open("/tmp/news_model_" + currency_pair, 'wb'))

		pickle.dump(kmeans, open("/tmp/news_kmeans_" + currency_pair, 'wb'))

		continue
		'''

		clf = GradientBoostingRegressor(random_state=42)
		clf.fit(X_train, y_train_mean)


		test_calendar = get_calendar_df(test_pair, year)

		prices, times = load_time_series(test_pair, None)

		price_df = pd.DataFrame()
		price_df['prices'] = prices
		price_df['times'] = times

				
		equity_curve = back_test(first_currency, second_currency, price_df, test_pair, year,test_calendar, clf)

		if test_pair[4:7] != "JPY":
			total_pnl += equity_curve[-1] * 100
		else:
			total_pnl += equity_curve[-1]

		print "Total Profit", total_pnl

		import datetime
		import numpy as np
		from matplotlib.backends.backend_pdf import PdfPages
		import matplotlib.pyplot as plt


		with PdfPages('/tmp/equity_curve_' + str(year) + '_' + test_pair + '.pdf') as pdf:
			plt.figure(figsize=(6, 4))
			plt.plot(equity_curve)

			pdf.savefig()  # saves the current figure into a pdf page
			plt.close() 

