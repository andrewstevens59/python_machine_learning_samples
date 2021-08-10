


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
        self.max_profit = 0


def get_order_book(symbol, time, curr_price):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(pycurl.ENCODING, 'gzip') 
	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v3/instruments/USD_JPY/orderBook?time=" + time + "&bucketWidth=0.0005")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8', 'Content-Type: application/json'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['orderBook']['buckets']

	print "-------------"

	curr_spreads = []
	for item in j:
		curr_spreads.append(abs(float(item['price']) - curr_price))


	std = np.std(curr_spreads)
	mean = np.mean(curr_spreads)

	num = 0
	denom = 0
	for item in j:

		if abs(float(item['price']) - curr_price) > 0:
			wt = 1.0 / abs(float(item['price']) - curr_price) 
			num +=  ((float(item['longCountPercent']) - float(item['shortCountPercent'])) / (1)) * wt
			denom += wt

	return num

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

	returns = []

	pos_profit = 0
	curr_equity = 50000

	is_start = False

	index = 0
	max_equity = 50000
	while index < len(j):
		item = j[index]

		times.append(item['time'])
		prices.append([item['closeMid']])
		volumes.append([item['volume']])


		net_sum = get_order_book(symbol, item['time'], prices[-1][0])

		before = curr_equity

		if index < len(j) - 48:

			avg_return = np.mean(labels[-4:])

			if len(labels) > 4:
				if (avg_return > 0) != (net_sum > 0):

					pnl = j[index + 47]['closeMid'] - j[index]['closeMid']

					if (net_sum < 0) == (pnl > 0):
						curr_equity += abs(pnl) * abs(net_sum) * 10000 * abs(avg_return)
					else:
						curr_equity += -abs(pnl) * abs(net_sum) * 10000 * abs(avg_return)


			rates.append(j[index + 47]['closeMid'] - j[index]['closeMid'])
			labels.append(j[index + 47]['closeMid'] - j[index]['closeMid'])

			if (net_sum > 0) != (labels[-1] > 0):
				pnl = abs(labels[-1]) * 10000
			else:
				pnl = -abs(labels[-1]) * 10000

			

			print pnl, "***", max(1, abs(curr_equity - max_equity))

			print curr_equity



			max_equity = max(max_equity, curr_equity)

	
			'''
			if pnl < 0:

				if is_start:
					curr_equity += pnl * abs(net_sum)

				is_start = True
			else:

				if is_start:
					curr_equity += pnl * abs(net_sum)

					print "neg"

				is_start = False
			'''



		index += 48
		returns.append(curr_equity - before)
				

	return rates, prices, labels, price_range, times, volumes, returns


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



def back_test_strategy(times, martingale_type, entry_bias, trade_dir, biased_predictions, test_prices, pip_size, commission, pair):

	first_currency = pair[0:3]
	second_currency = pair[4:7]


	orders = []
	net_order_amount = []
	time_stamps = []
	avg_hold_times = []
	order_invert_nums = []
	num_orders = 0

	returns = []
	order_amount = 0
	curr_equity = 5000
	curr_float_profit = 0
	buy_num = 1
	sell_num = 1


	wait_time = 2
	reduce_order = 0
	hold_time = 6
	net_pip_diff = 0
	prev_dir1 = None
	prev_dir2 = None
	threshold_cutoff = 2
	trade_num = 0

	equity_factor = 1
	equity_factors = []
	equity_curve = []
	orders_open = []
	closed_order_profit = []
	closed_order_times = []

	replay_trades = []

	draw_downs = []
	max_equity = 0

	mean_price = np.mean(test_prices)

	multiplier = 1
	entry_prediction = 0
	change_side_count = 0

	max_order_num = 0
	for index in range(len(biased_predictions)):

		biased_prediction = biased_predictions[index] * trade_dir
		curr_price = test_prices[index][0]

		time = times[index]

		order_num = 0
		max_open_time = 0
		for order in orders:
			max_open_time = max(max_open_time, order.open_time)
			if (order.dir == True) == (biased_prediction > 0):
				order_num += 1

		is_close_all = False
		
		
		if (biased_prediction * trade_dir < 0) != prev_dir1:
			entry_bias = 2.5
			hold_time = 0
			change_side_count += 1
		

		buy_prices = []
		sell_prices = []
		max_buy_profit = -99999999
		max_sell_profit = -99999999
		buy_amount = 0
		sell_amount = 0

		buy_profit = 0
		sell_profit = 0
		net_pip_diff = 0
		float_profit = 0
		avg_hold_time = 0
		total_amount = 0

		buy_num = 0
		sell_num  = 0

		pos_pnl_orders = []
		neg_pnl_orders = []
		for order in orders:

			if (order.dir == True) == (curr_price > order.open_price):
				pip_diff = abs(curr_price - order.open_price) / pip_size
				profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.amount
			else:
				pip_diff = -abs(curr_price - order.open_price) / pip_size
				profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.amount


			net_pip_diff += profit
			float_profit += profit
			avg_hold_time += (index - order.open_time) 
			total_amount += order.amount

			order.pnl = profit

			if order.pnl > 0:
				pos_pnl_orders.append(order)
			else:
				neg_pnl_orders.append(order)

			if order.dir == True:
				buy_prices.append(order.open_price)
				max_buy_profit = max(max_buy_profit, profit)
				buy_amount += order.amount
				buy_profit += profit
				buy_num += 1
			else:
				sell_prices.append(order.open_price)
				max_sell_profit = max(max_sell_profit, profit)
				sell_amount += order.amount
				sell_profit += profit
				sell_num += 1

		pos_pnl_orders = sorted(pos_pnl_orders, key=lambda x: x.pnl, reverse=True)
		neg_pnl_orders = sorted(neg_pnl_orders, key=lambda x: x.pnl, reverse=True)

		if len(pos_pnl_orders) > 0 and len(neg_pnl_orders) > 0:
			total_profit = pos_pnl_orders[0].pnl + neg_pnl_orders[0].pnl
		else:
			total_profit = 0


		if len(orders) > 0:
			avg_hold_time /= len(orders)

		if (abs(biased_prediction) >= 2.0):

			for buy_sell in [True, False]:

				if buy_sell != biased_prediction > 0 or buy_sell == (buy_amount > sell_amount):
					continue

				entry_bias = abs(biased_prediction) + 0.1
				order = Order()


				if buy_sell:
					max_profit = max_buy_profit
					order.amount = (sell_amount) * 2
				else:
					max_profit = max_sell_profit
					order.amount = (buy_amount) * 2

				order.amount = max(1, max(sell_amount, buy_amount) * 2)

				if max_profit > 0:
					continue

				order.dir = buy_sell
				order.open_time = index
				order.open_price = curr_price
				
				order.open_predict = biased_prediction

				change_side_count = 0
				
				if len(equity_factors) > 0:
					order.equity_factor = 1
				else:
					order.equity_factor = 1
			

				orders.append(order)
				num_orders += 1

				prev_dir1 = biased_prediction * trade_dir

				reduce_order += 0.1
				trade_num += 1

		float_profit = 0
		prev_equity = curr_equity
		prev_float_profit = curr_float_profit
		prev_order_amount = order_amount
		prev_pip_diff = net_pip_diff



		order_amount = 0
		is_close = False
		found_match = 	False

		buy_profit = 0
		sell_profit = 0
		for order in orders:

			if (order.dir == True) == (curr_price > order.open_price):
				pip_diff = abs(curr_price - order.open_price) / pip_size
				profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.amount
			else:
				pip_diff = -abs(curr_price - order.open_price) / pip_size
				profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.amount

			if order.dir:
				buy_profit += profit
			else:
				sell_profit += profit

			float_profit += profit
			order_amount += abs(order.amount)


		if len(orders) > 0:

			'''
			history_prices = test_prices[max(0, index-40):index + 1]
			kmeans = KMeans(n_clusters=min(5, len(history_prices)), init='k-means++', max_iter=100, n_init=1, 
	                               random_state = 42).fit(history_prices)

			prediction = kmeans.predict([history_prices[-1]])[0]
			mean_center = kmeans.cluster_centers_.tolist()[prediction][0]

			if pair[4:7] == "JPY":
				mult = 100
			else:
				mult = 1
			'''

			new_orders = []
			is_found = False
			for order in orders:

				pip_diff = ((abs(curr_price - order.open_price) - commission) / pip_size)
				if (order.dir == True) == (curr_price > order.open_price):
					profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.amount
				else:
					profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.amount

				'''
				if profit < 0 and abs(curr_price - mean_center) > 0.02 * mult:
					float_profit -= profit 
					curr_equity += profit * order.equity_factor
					closed_order_profit.append(profit)

					is_found = True
					continue
				'''
 
				if (abs(biased_prediction) >= 2.0) and (biased_prediction > 0) != order.dir and profit > 0:
					float_profit -= profit 
					curr_equity += profit * order.equity_factor
					closed_order_profit.append(profit)
					closed_order_times.append(len(biased_predictions) - index)

					is_found = True
					continue
				
		
				if (len(orders) > 0 and orders[-1].amount >= 16) or (index >= len(biased_predictions)  - 1):

					float_profit -= profit 
					curr_equity += profit * order.equity_factor
					closed_order_profit.append(profit)
					closed_order_times.append(len(biased_predictions) - index)

			
					is_found = True
					continue
				

				new_orders.append(order)

			orders = new_orders

			max_order_num = max(max_order_num, len(orders))

		curr_float_profit = 0
		for order in orders:
			if (order.dir == True) == (curr_price > order.open_price):
				profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.amount
			else:
				profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.amount

			curr_float_profit += profit * order.equity_factor

		equity_factor = (curr_equity + curr_float_profit) / 5000
		equity_factors.append(equity_factor)

		equity_curve.append(curr_equity + curr_float_profit)
		orders_open.append(len(orders))
		

		returns.append((curr_equity + curr_float_profit) - (prev_equity + prev_float_profit))
		net_order_amount.append(order_amount - prev_order_amount)

		max_equity = max(max_equity, curr_equity + curr_float_profit)
		draw_down = 0#1 - (float(curr_equity + curr_float_profit) / max_equity)
		draw_downs.append(draw_down)

	avg_return = curr_equity + curr_float_profit - 5000

	rating = (((curr_equity + curr_float_profit) / 5000) - 1) / max(0.01, max(draw_downs))

	#print curr_equity, "        ", (max_order_num / 2), "***"

	return equity_curve, returns, rating, max(draw_downs), np.mean(closed_order_profit), closed_order_profit, replay_trades, closed_order_times

def get_portfolio_series(norm_prices):

	rates = []
	prices = []
	labels = []
	times = []
	price_range = []
	volumes = []

	for index in range(len(norm_prices)):
	    price = norm_prices[index]

	    prices.append([price])

	    if index < len(norm_prices) - 48:
			labels.append(norm_prices[index + 47] - norm_prices[index])

	return prices, labels

def get_basket_series(norm_prices, actual_prices):

	if len(norm_prices) != len(actual_prices):
		print "order size"
		sys.exit(0)

	rates = []
	prices = []
	labels = []
	times = []
	price_range = []
	volumes = []

	for index in range(len(norm_prices)):
	    price = norm_prices[index]

	    prices.append([price])

	    if index < len(norm_prices) - 48:
			labels.append(actual_prices[index + 47] - actual_prices[index])

	return prices, labels
	
def find_market_prices(actual_prices, labels):

	start = 0
	end = 700
	predictions = []
	current_prices = []
	while end < len(labels):
		current_prices.append([actual_prices[end - 1]])
		start += 12
		end += 12

	return current_prices

def train_and_back_test_stat_arb(trade_logger, model_key, pair, is_use_residual, martingale_type, global_currency_pairs, avg_spreads, avg_prices, entry_bias, trade_dir, is_train_model, root_dir):

	print model_key, root_dir

	trade_logger.info('Model Key: ' + model_key) 

	norm_prices = pickle.load(open(root_dir + "stat_arb_series_" + model_key, 'rb'))
	times = {model_key : [1] * len(norm_prices[model_key])}
	portfolio_wts = pickle.load(open(root_dir + "portfolio_wts_" + model_key, 'rb'))

	if "_basket" in model_key:
		portfolio_pairs = [pair]
		actual_prices = pickle.load(open(root_dir + "actual_price_series_" + model_key, 'rb'))
		prices1, y1 = get_basket_series(norm_prices[model_key], actual_prices[model_key])

		if pair[4:7] == 'JPY':
			pip_size = 0.01
		else:
			pip_size = 0.0001

		commission = avg_spreads[pair] * pip_size

	else:
		portfolio_pairs = portfolio_wts[model_key]['currency_pairs']
		prices1, y1 = get_portfolio_series(norm_prices[model_key])

		commission = 0
		for currency_pair in portfolio_wts[model_key]['wt']:

			if currency_pair[4:7] == 'JPY':
				pip_size = 0.01
			else:
				pip_size = 0.0001

			commission += abs(portfolio_wts[model_key]['wt'][currency_pair]) * (avg_spreads[currency_pair] * pip_size)

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


	whitelist = []
	for model, lag in zip(x_sets, lags):

		model_key_base = model_key + "_" + str(model.__class__.__name__) + "_" + str(lag)

		if lag == 0:
			predictions, test_prices = model.init(root_dir, model_key, None, prices1, y1, None, is_use_residual, is_train_model)
		else:
			predictions, test_prices = model.init(root_dir, model_key, None, prices1, y1, None, lag, is_use_residual, is_train_model)

		if "_basket" in model_key:
			test_prices = find_market_prices(actual_prices[model_key], y1)

		if predictions == None:
			print "not found", model_key_base
			continue

		equity_curve, returns, rating, draw_down, avg_profit, closed_profits, replay_trades, closed_times = back_test_strategy(times[model_key], martingale_type, entry_bias, trade_dir, predictions, test_prices, 1, commission, model_key)
		
		if len(closed_profits) > 1:
			sharpe_all = (float(np.mean(closed_profits)) / np.std(closed_profits)) 
		else:
			sharpe_all = 0 

		from sklearn.linear_model import LinearRegression
		reg = LinearRegression().fit([[v] for v in range(len(equity_curve))], equity_curve)
		y_pred = reg.predict([[v] for v in range(len(equity_curve))])

		r2 = r2_score(equity_curve, y_pred)

		trade_logger.info(str(model_key_base) + " " + str(sharpe_all) + " " + str(r2) + " " + str(portfolio_wts))

		if sharpe_all < 0.0 and False:
			equity_curve = None
		else:
			print model_key_base, sharpe_all, r2, np.median(closed_profits), np.mean(closed_profits)

		whitelist.append({"is_use_residual" : is_use_residual, "closed_times" : closed_times, "closed_profits" : closed_profits, "replay_trades" : replay_trades, "prediction_key" : model_key, "model_key" : model_key_base, "wts" : portfolio_wts[model_key]['wt'], "currency_pairs" : portfolio_pairs, "num_trades" : len(closed_profits), "r2_score" : r2, "martingale_type" : martingale_type, "draw_down" : draw_down, "avg_profit" : avg_profit, "equity_curve" : equity_curve, "sharpe" : sharpe_all, "entry_bias" : entry_bias, "trade_dir" : trade_dir})


	return whitelist

