


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
        self.open_price1 = 0
        self.open_price2 = 0
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

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, returns):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return p_var

def max_sharpe_ratio(returns, optimization_bounds = (-1.0, 1.0)):

    df = pd.DataFrame(returns)
    mean_returns = df.mean()
    cov_matrix = df.cov().values.tolist()

    print df.corr()

    weights = [0.001 / cov_matrix[0][0], 0.001 / cov_matrix[1][1]]
    weight_sum = abs(weights[0]) + abs(weights[1])
    #weights = np.array([w / weight_sum for w in weights])

    if cov_matrix[0][1] < 0:
    	weights[1] = -weights[1]

    return weights, df.corr().values.tolist()[0][1]

def get_calendar_df(pair, year): 

	if pair != None:
		currencies = set([pair[0][0:3], pair[0][4:7], pair[1][0:3], pair[1][4:7]])
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



def create_training_set(price_df, pair, year, test_calendar, description_map):

	X_train = []
	y_train = []

	currencies = set([pair[0][0:3], pair[0][4:7], pair[1][0:3], pair[1][4:7]])

	prev_time = None
	for index, row in test_calendar.iterrows():

		if row['time'] == prev_time:
			continue

		prev_time = row['time']

		before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * 4).values.tolist()[0]

		future_prices = (price_df[price_df['times'] >= row['time']])['prices'].head(24 * 5 * 2).values.tolist()

		if len(future_prices) < 12:
			continue

		calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= row["time"])]

		feature_vector = []
		for currency in currencies:

			calendary_currency = calendar_history[calendar_history["currency"] == currency]
			description_vector1 = [0] * len(description_map)
			description_vector2 = [0] * len(description_map)
			for index, history in calendary_currency.iterrows():
				description_vector1[description_map[history["description"]]] = history['actual'] - history['forecast']
				description_vector2[description_map[history["description"]]] = history['actual'] - history['previous']

			feature_vector += description_vector1 + description_vector2

		X_train.append(feature_vector)
		y_train.append(future_prices[-1] - future_prices[0])

	return X_train, y_train

def back_test(price_df, pair, year, test_calendar, model_mean, description_map, wts):

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	if pair[0][4:7] == 'JPY':
		pip_size = 0.01
	else:
		pip_size = 0.0001

	commission1 = avg_spreads[pair[0]] * pip_size
	commission2 = avg_spreads[pair[1]] * pip_size
	currencies = set([pair[0][0:3], pair[0][4:7], pair[1][0:3], pair[1][4:7]])

	print "no"

	equity = 0
	equity_curve = []
	orders = []
	returns = []

	end_of_year_time = test_calendar["time"].tail(1).values.tolist()[0]

	anchor_price = None
	prev_time = None
	last_open_time = -99999
	for index, row in test_calendar.iterrows():

		if row["time"] == prev_time:
			continue

		prev_time = row["time"]

		#print price_df[price_df['times'] >= row['time']]

		before_time = (price_df[price_df['times'] <= row['time']])['times'].tail(24 * 5 * 4).values.tolist()[0]
		future_times = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['times'].values.tolist()[12:]
		future_prices = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['prices'].values.tolist()[12:]


		future_indv_price1 = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['price1'].values.tolist()[12:]
		future_indv_price2 = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['price2'].values.tolist()[12:]
		future_indv_mult1 = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['mult_factor_0'].values.tolist()[12:]
		future_indv_mult2 = (price_df[(price_df['times'] >= row['time']) & (price_df["times"] <= end_of_year_time)])['mult_factor_1'].values.tolist()[12:]

		if len(future_prices) < 12:
			continue

		calendar_history = test_calendar[(test_calendar["time"] >= before_time) & (test_calendar["time"] <= row["time"])]

		feature_vector = []
		for currency in currencies:

			calendary_currency = calendar_history[calendar_history["currency"] == currency]
			description_vector1 = [0] * len(description_map)
			description_vector2 = [0] * len(description_map)
			for index, history in calendary_currency.iterrows():
				description_vector1[description_map[history["description"]]] = history['actual'] - history['forecast']
				description_vector2[description_map[history["description"]]] = history['actual'] - history['previous']

			feature_vector += description_vector1 + description_vector2

		first_price = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]
		first_time = (price_df[price_df['times'] >= row['time']])['prices'].head(1).values.tolist()[0]

		if anchor_price == None:
			anchor_price = first_price

		anchor_price = first_price
		anchor_time = first_time

		mean_prediction = model_mean.predict([feature_vector])[0]

		max_total_pnl = equity
		max_delta = 0
		found_news_release = False
		time_step = 11

		curr_calendar_time = row["time"]
		for future_index in range(len(future_prices)):
			future_price = future_prices[future_index]
			future_time = future_times[future_index]

			mult_factor1 = future_indv_mult1[future_index] * wts[0]
			mult_factor2 = future_indv_mult2[future_index] * wts[1]

			indv_price1 = future_indv_price1[future_index]
			indv_price2 = future_indv_price2[future_index]

			time_step += 1

			if abs(future_price - anchor_price) > max_delta * 2:
				delta = ((future_price - first_price) - mean_prediction) / 1

				new_order = Order()
				new_order.open_price = future_price
				new_order.open_price1 = indv_price1
				new_order.open_price2 = indv_price2
				new_order.open_time = future_time
				last_open_time = time_step
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
					if (indv_price1 > order.open_price1) == (order.dir):
						equity += (abs(indv_price1 - order.open_price1) - commission1) * order.amount * mult_factor1
					else:
						equity += (-abs(indv_price1 - order.open_price1) - commission1) * order.amount * mult_factor1

					if (indv_price2 > order.open_price2) != (order.dir):
						equity += (abs(indv_price2 - order.open_price2) - commission2) * order.amount * mult_factor2
					else:
						equity += (-abs(indv_price2 - order.open_price2) - commission2) * order.amount * mult_factor2
					continue

				if (indv_price1 > order.open_price1) == (order.dir):
					total_pnl += (abs(indv_price1 - order.open_price1) - commission1) * order.amount * mult_factor1
				else:
					total_pnl += (-abs(indv_price1 - order.open_price1) - commission1) * order.amount * mult_factor1

				if (indv_price2 > order.open_price2) != (order.dir):
					total_pnl += (abs(indv_price2 - order.open_price2) - commission2) * order.amount * mult_factor2
				else:
					total_pnl += (-abs(indv_price2 - order.open_price2) - commission2) * order.amount * mult_factor2

				order_count += 1

			orders = new_orders

			if (abs(future_price - anchor_price) < max_delta / (time_step * 0.1)):
				equity += total_pnl
				total_pnl = 0
				max_total_pnl = 0
				max_delta = 0
				last_open_time = -99999
				anchor_price = first_price

				for order in orders:

					if (future_price > order.open_price) == (order.dir):
						returns.append(abs(future_price - order.open_price) * order.amount)
					else:
						returns.append(-abs(future_price - order.open_price) * order.amount)

				orders = []
				break


		if found_news_release == False or True:
			anchor_price = None

			equity += total_pnl
			for order in orders:

				if (future_price > order.open_price) == (order.dir):
					returns.append(abs(future_price - order.open_price) * order.amount)
				else:
					returns.append(-abs(future_price - order.open_price) * order.amount)

			orders = []

		equity_curve.append(equity)

	return equity_curve, returns


def create_data_set(year_range, currency_pair, select_year, description_map, wts):

	X_train = []
	y_train = []

	for year in year_range:
		if year == select_year:
			continue

		print "Year", year
		
		for test_pair in [currency_pair]:

			prices, times = load_time_series(currency_pair[0], None)
			price_df1 = pd.DataFrame()
			price_df1['price1'] = prices
			price_df1['times'] = times


			prices, times = load_time_series(currency_pair[1], None)
			price_df2 = pd.DataFrame()
			price_df2['price2'] = prices
			price_df2['times'] = times

			price_df = price_df1.set_index('times').join(price_df2.set_index('times'))
			price_df.reset_index(inplace=True)

			for curr_pair, pair_index in zip(currency_pair, range(2)):
				if curr_pair[4:7] == "USD":
					price_df['mult_factor_' + str(pair_index)] = 1.0
				elif curr_pair[4:7] + "_USD" in currency_pairs:
					prices, times = load_time_series(curr_pair[4:7] + "_USD", None)

					conv_df = pd.DataFrame()

					conv_df['mult_factor_' + str(pair_index)] = prices
					conv_df['times'] = times
					price_df = price_df.set_index('times').join(conv_df.set_index('times'))
					price_df.reset_index(inplace=True)
				else:
					prices, times = load_time_series("USD_" + curr_pair[4:7], None)

					conv_df = pd.DataFrame()
					conv_df['mult_factor_' + str(pair_index)] = [1.0 / p for p in prices]
					conv_df['times'] = times
					price_df = price_df.set_index('times').join(conv_df.set_index('times'))
					price_df.reset_index(inplace=True)

			price_df["prices"] = price_df.apply(lambda x: (x["price1"] * x["mult_factor_0"] * wts[0]) - (x["price2"] * x["mult_factor_1"] * wts[1]), axis=1)

			test_calendar = get_calendar_df(test_pair, year)

			x, y = create_training_set(price_df, test_pair, year, test_calendar, description_map)

			X_train += x
			y_train += y


	return X_train, y_train

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

#["GBP_AUD", "GBP_CAD"]


import sqlite3

conn = sqlite3.connect('/Users/callummc/news_stat_arb.db')
print "Opened database successfully";


conn.execute('''
	CREATE TABLE IF NOT EXISTS BACKTEST_DATA
         (
         PAIR           CHAR(50),
         PROFIT      FLOAT,
         YEAR  CHAR(50),
         CORRELATION      FLOAT,
         CONSTRAINT UC_Person UNIQUE (PAIR,YEAR)
         );
         ''')

print "Table created successfully";


conn = sqlite3.connect('/Users/callummc/news_stat_arb.db')
cursor=conn.cursor()
cursor.execute("SELECT sum(PROFIT) FROM BACKTEST_DATA WHERE CORRELATION>0.5 group by YEAR")
data=cursor.fetchall()
print "Year", data

cursor.execute("SELECT sum(PROFIT) FROM BACKTEST_DATA WHERE CORRELATION>0.5")
data=cursor.fetchall()
print "Total", data

cursor.execute("SELECT PAIR FROM BACKTEST_DATA group by PAIR having min(PROFIT) > 0")
data=cursor.fetchall()
print "All Positive", data
#cursor.execute('DELETE FROM BACKTEST_DATA where PAIR = "' + str(["EUR_USD", "AUD_USD"]) + '"')
#conn.commit()

conn.close()


pairs_list = []
for currency_pair1 in range(len(currency_pairs)):
	for currency_pair2 in range(len(currency_pairs)):
		if currency_pair1 < currency_pair2:
			pairs_list.append([currency_pairs[currency_pair1], currency_pairs[currency_pair2]])

for currency_pair in [["EUR_JPY", "USD_JPY"]]:
	print currency_pair

	total_pnl = 0

	prices, times = load_time_series(currency_pair[0], None)
	price_df1 = pd.DataFrame()
	price_df1['price1'] = prices
	price_df1['times'] = times
	prices, times = load_time_series(currency_pair[1], None)
	price_df2 = pd.DataFrame()
	price_df2['price2'] = prices
	price_df2['times'] = times
	price_df = price_df1.set_index('times').join(price_df2.set_index('times'))
	price_df["diff1"] = price_df["price1"].diff(periods=24*5*2)
	price_df["diff2"] = price_df["price2"].diff(periods=24*5*2)
	price_df.reset_index(inplace=True)

	for curr_pair, pair_index in zip(currency_pair, range(2)):
		if curr_pair[4:7] == "USD":
			price_df['mult_factor_' + str(pair_index)] = 1.0
		elif curr_pair[4:7] + "_USD" in currency_pairs:
			prices, times = load_time_series(curr_pair[4:7] + "_USD", None)

			conv_df = pd.DataFrame()

			conv_df['mult_factor_' + str(pair_index)] = prices
			conv_df['times'] = times
			price_df = price_df.set_index('times').join(conv_df.set_index('times'))
			price_df.reset_index(inplace=True)
		else:
			prices, times = load_time_series("USD_" + curr_pair[4:7], None)

			conv_df = pd.DataFrame()
			conv_df['mult_factor_' + str(pair_index)] = [1.0 / p for p in prices]
			conv_df['times'] = times
			price_df = price_df.set_index('times').join(conv_df.set_index('times'))
			price_df.reset_index(inplace=True)

	price_df["diff1"] = price_df.apply(lambda x: x["mult_factor_0"] * x["diff1"], axis=1)
	price_df["diff2"] = price_df.apply(lambda x: x["mult_factor_1"] * x["diff2"], axis=1)

	wts, corr = max_sharpe_ratio(zip(price_df["diff1"], price_df["diff2"]))
	'''
	if abs(corr) < abs(0.65):
		continue
	'''


	print wts

	description_map = {}
	for year in range(2007, 2018):
		descriptions = get_calendar_df(currency_pair, year)["description"].values.tolist()
		for description in descriptions:
			if description not in description_map: 
				description_map[description] = len(description_map)

	for year in range(2008, 2018):

		conn = sqlite3.connect('/Users/callummc/news_stat_arb.db')
		cursor=conn.cursor()
		cursor.execute("SELECT * FROM BACKTEST_DATA WHERE PAIR = ? AND YEAR = ?", (str(currency_pair),str(year)))
		data=cursor.fetchall()
		if len(data) > 0:
			print "Already Exists"
			conn.close()
			continue

		X_train, y_train = create_data_set(range(year-1, year), currency_pair, year, description_map, wts)
 
		#pickle.dump(X_train, open("/tmp/X_train", 'wb'))
		#pickle.dump(y_train, open("/tmp/y_train", 'wb'))
		#pickle.dump(kmeans, open("/tmp/kmeans", 'wb'))


		for test_pair in [currency_pair]:
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

			clf_mean = GradientBoostingRegressor(random_state=42)
			clf_mean.fit(X_train, y_train)

			test_calendar = get_calendar_df(test_pair, year)

			prices, times = load_time_series(test_pair[0], None)
			price_df1 = pd.DataFrame()
			price_df1['price1'] = prices
			price_df1['times'] = times


			prices, times = load_time_series(test_pair[1], None)
			price_df2 = pd.DataFrame()
			price_df2['price2'] = prices
			price_df2['times'] = times

			price_df = price_df1.set_index('times').join(price_df2.set_index('times'))
			price_df.reset_index(inplace=True)

			diff_series = zip(price_df["price1"].values.tolist(), price_df["price2"].values.tolist())


			for curr_pair, pair_index in zip(test_pair, range(2)):
				if curr_pair[4:7] == "USD":
					price_df['mult_factor_' + str(pair_index)] = 1.0
				elif curr_pair[4:7] + "_USD" in currency_pairs:
					prices, times = load_time_series(curr_pair[4:7] + "_USD", None)

					conv_df = pd.DataFrame()

					conv_df['mult_factor_' + str(pair_index)] = prices
					conv_df['times'] = times
					price_df = price_df.set_index('times').join(conv_df.set_index('times'))
					price_df.reset_index(inplace=True)
				else:
					prices, times = load_time_series("USD_" + curr_pair[4:7], None)

					conv_df = pd.DataFrame()
					conv_df['mult_factor_' + str(pair_index)] = [1.0 / p for p in prices]
					conv_df['times'] = times
					price_df = price_df.set_index('times').join(conv_df.set_index('times'))
					price_df.reset_index(inplace=True)

			price_df["prices"] = price_df.apply(lambda x: (x["price1"] * x["mult_factor_0"] * wts[0]) - (x["price2"] * x["mult_factor_1"] * wts[1]), axis=1)

			equity_curve, returns = back_test(price_df, test_pair, year, test_calendar, clf_mean, description_map, wts)

			conn.execute('insert into BACKTEST_DATA values (?,?,?,?)', [str(currency_pair), float(equity_curve[-1]), str(year),abs(corr)])
			conn.commit()
			conn.close()

			total_pnl += equity_curve[-1] 

			print "Sharpe", np.mean(returns) / np.std(returns)
			print "Total Profit", total_pnl


