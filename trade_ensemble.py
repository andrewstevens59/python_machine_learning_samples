

import sys
import math
from datetime import datetime, date, time
from random import *
import os.path


import pickle

import pycurl
from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from uuid import getnode as get_mac
import socket

import time
import calendar
import json
import copy

import pickle
import math
import sys

import numpy as np
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
from datetime import date,timedelta
import train_and_back_test_all as back_test_all
import plot_equity as portfolio
from maximize_sharpe import *
from close_trade import *

import delta_process as delta_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import json 

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]



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

	rates = []
	prices = []
	labels = []
	price_range = []

	for index in range(len(j)):
		item = j[index]


		rates.append([item['highMid'] - item['lowMid'], item['closeMid'] - item['openMid']])
		prices.append([item['closeMid']])
		price_range.append(item['closeMid'] - item['openMid'])

		if index < len(j) - 96:
			labels.append(j[index + 95]['closeMid'] - j[index + 1]['openMid'])

	return rates, prices, labels, price_range



def initiate_trades(buy_account_number, sell_account_number):

	response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + buy_account_number + "/", "GET")
	j = json.loads(response_value)
	profit = float(j['account'][u'unrealizedPL'])
	buy_balance = float(j['account'][u'balance'])
	margin_used = float(j['account'][u'marginUsed'])

	print "Buy Balance: ", buy_balance, "Profit: ", profit, "Marin Used: ", margin_used

	response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + sell_account_number + "/", "GET")
	j = json.loads(response_value)
	profit = float(j['account'][u'unrealizedPL'])
	sell_balance = float(j['account'][u'balance'])
	margin_used = float(j['account'][u'marginUsed'])

	print "Sell Balance: ", sell_balance, "Profit: ", profit, "Marin Used: ", margin_used

	balance = buy_balance + sell_balance

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))
	model_pair_wt = pickle.load(open("/tmp/model_pair_wt", 'rb'))

	buy_pair_time_diff = close_trades(buy_account_number)
	sell_pair_time_diff = close_trades(sell_account_number)

	if get_mac() == 154505288144005:
		remove_model_pair = set()
	else:
		remove_model_pair = pickle.load(open("/tmp/remove_model_pair", 'rb'))

	times = []
	global_returns = []
	global_exposures = []
	global_currency_pairs = [] 
	for pair in currency_pairs:

		print pair

		pip_size = 0.0001

		if pair[4:] == "JPY":
			pip_size = 0.01

		remove_model_pair, global_exposures, global_returns, times, global_currency_pairs, final_prediction = back_test_all.train_and_back_test(pair, remove_model_pair, model_pair_wt, global_currency_pairs, global_exposures, global_returns, times, avg_spreads, avg_prices)

		print "Prediction", final_prediction

		order_amount = int(round(abs(final_prediction) * balance * 15))
		if order_amount == 0:
			continue

		buy_pair_time_diff = close_trades(buy_account_number)
		sell_pair_time_diff = close_trades(sell_account_number)

		if pair in buy_pair_time_diff and buy_pair_time_diff[pair] < 6:
			continue

		if pair in sell_pair_time_diff and sell_pair_time_diff[pair] < 6:
			continue

		# 20 is 20% leverage * 4 trades in 2 days - 1 every 12 hours

		curr_price = 0
		min_price = 999999
		max_price = 0
		for spread_count in range(5):

			response = sendCurlRequest("https://api-fxtrade.oanda.com/v1/prices?instruments=" + pair, "GET")
			response = json.loads(response)['prices']

			spread = 0
			for price in response:
				if price['instrument'] == pair:
					curr_price = (price['bid'] + price['ask']) / 2
					spread = abs(price['bid'] - price['ask']) / pip_size
					break

			if curr_price == 0:
				print "price not found"
				continue

			min_price = min(min_price, price['ask'])
			max_price = max(max_price, price['bid'])

			if final_prediction > 0:
				spread -= max(0, (max_price - price['ask']) / pip_size)
			else:
				spread -= max(0, (price['bid'] - min_price) / pip_size)


			print spread, avg_spreads[pair], order_amount
			
			if spread <= avg_spreads[pair]:
				break 

			time.sleep(5) 

		if spread > avg_spreads[pair]:
			# add a pending order
			print "Adding Pending Order"

			pending_trade = {"amount" : order_amount, "price" : curr_price, 
				"spread" : avg_spreads[pair], "pair" : pair, "open_time" : time.time(), 
				"prediction" : final_prediction}

			if final_prediction > 0:
				file = "/tmp/buy_" + pair + ".pending_order"
			else:
				file = "/tmp/sell_" + pair + ".pending_order"
			pickle.dump(pending_trade, open(file, 'wb'))

			if get_mac() == 154505288144005:
				import paramiko
				print "transferring"
				t = paramiko.Transport(("158.69.218.215", 22))
				t.connect(username="root", password="jEC1ZbfG")
				sftp = paramiko.SFTPClient.from_transport(t)
				sftp.put(file, file)

			continue 

		print pair, order_amount, balance


		precision = '%.4f'
		if pair[4:] == 'JPY':
			precision = '%.3f'

		if final_prediction > 0:
			tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (300 * pip_size))) + '"}'
			sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price - (300 * pip_size))) + '"}'
			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + buy_account_number + "/orders", "POST", '{ "order" : {' + tp_price + ', ' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
		else:
			tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price - (300 * pip_size))) + '"}'
			sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (300 * pip_size))) + '"}'
			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + sell_account_number + "/orders", "POST", '{ "order" : {' + tp_price + ', ' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		
	
	return remove_model_pair, global_returns, times, global_currency_pairs, final_prediction

#collect_avg_pip_size()
	

remove_model_pair, global_returns, times, global_currency_pairs, final_prediction = initiate_trades("001-001-1370090-004", "001-001-1370090-003")



if get_mac() == 154505288144005:
	wts, global_currency_pairs, equity = portfolio.calculate_portfolio(global_returns, global_currency_pairs, times)

	model_pair_wt = {}
	for wt, pair in zip(wts, global_currency_pairs):
		model_pair_wt[pair] = wt

	pickle.dump(model_pair_wt, open("/tmp/model_pair_wt", 'wb'))

	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.put("/tmp/model_pair_wt", "/tmp/model_pair_wt")






