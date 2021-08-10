import sys
import math
from datetime import datetime
from random import *
import os.path


import pickle

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
import string
import random as rand

import os
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
from uuid import getnode as get_mac
import socket
import paramiko
import json

import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import logging
from close_trade import *
import os



class Order:

	def __init__(self):
		self.pair = ""
		self.dir = 0
		self.portfolio_dir = 0
		self.open_price = 0
		self.open_time = 0
		self.amount = 0
		self.PnL = 0
		self.tp_price = 0
		self.sl_price = 0
		self.actual_amount = 0
		self.account_number = None
		self.time_diff_hours = 0
		self.order_id = 0
		self.base_amount = 0
		self.sequence_id = 0
		self.model_key = None
		self.prediction_key = None
		self.margin_used = 0
		self.open_prediction = 0
		self.curr_prediction = 0
		self.portfolio_wt = 0


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

import datetime as dt

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

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def calculate_time_diff(s):
	week_day_start = datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").weekday()

	time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

	s = str(datetime.datetime.utcnow())
	s = s[0 : s.index('.')]

	week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

	time_end = time.time()

	time_diff_hours = time_end - time_start

	time_diff_hours /= 60 * 60

	week_num = int(time_diff_hours / (24 * 7))
	if week_num >= 1:
		time_diff_hours -= 48 * week_num

	if week_day_start != 6 and week_day_end < week_day_start:
		time_diff_hours -= 48
	elif week_day_end == 6 and week_day_end > week_day_start:
		time_diff_hours -= 48

	return time_diff_hours

def get_open_trades(account_number, order_metadata, total_margin):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades?count=50"

	while next_link != None:
		response_value, header_value = sendCurlRequest(next_link, "GET")

		lines = header_value.split('\n')
		lines = filter(lambda line: line.startswith('Link:'), lines)

		if len(lines) > 0:
			line = lines[0]
			next_link = line[line.find("<") + 1 : line.find(">")]
		else:
			next_link = None

		j = json.loads(response_value)
		open_orders = j['trades']
		open_orders = sorted(open_orders, key=lambda order: order['openTime'])

		for trade in open_orders:
			s = trade['openTime'].replace(':', "-")
			s = s[0 : s.index('.')]
			order_id = trade['id']
			open_price = float(trade[u'price'])
			pair = trade[u'instrument']
			amount = float(trade[u'currentUnits'])
			pair = trade['instrument'].replace("/", "_")
			PnL = float(trade['unrealizedPL'])
			margin_used = float(trade['marginUsed'])

			'''
			first_currency = pair[0:3]
			second_currency = pair[4:7]

			if first_currency != "USD":
				exposure_weight = avg_prices[first_currency + "_USD"]
			else:
				exposure_weight = 1

			if second_currency != "USD":
				pair_mult = avg_prices[second_currency + "_USD"]
			else:
				pair_mult = 1.0

			total_margin += float(margin_used)

			if amount > 0:
				price = avg_prices[pair + "_bid"]
			else:
				price = avg_prices[pair + "_ask"]

			if (amount > 0) == (avg_prices[pair] > open_price):
				print (abs(price - open_price) * abs(amount) * pair_mult), PnL, avg_prices[pair], pair_mult
			else:
				print (-abs(price - open_price) * abs(amount) * pair_mult), PnL, avg_prices[pair], pair_mult
			'''

			pip_size = 0.0001
			if pair[4:7] == "JPY":
				pip_size = 0.01

			time_diff_hours = calculate_time_diff(s)

			key = order_id + "_" + account_number
			
			if key not in order_metadata:
				order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
				trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
				print "not exist", key
				continue

			metadata = order_metadata[key]

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.time_diff_hours = time_diff_hours
			order.order_id = order_id
			order.account_number = account_number
			order.model_key = metadata["model_key"]
			order.open_prediction = metadata["open_prediction"]
			order.curr_prediction = metadata["curr_prediction"]
			order.dir = metadata["dir"]
			order.base_amount = metadata["base_amount"]
			order.sequence_id = metadata["sequence_id"]
			order.prediction_key = metadata["prediction_key"]
			order.portfolio_dir = metadata["portfolio_dir"]
			order.portfolio_wt = metadata["portfolio_wt"]

			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders, total_margin

def close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size):

	if (order.dir == True) == (curr_price > order.open_price):
		profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.base_amount
	else:
		profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.base_amount

	key = order.order_id + "_" + order.account_number
	if key in order_metadata:
		del order_metadata[key]

	existing_order_amount.remove(order.pair + "_" + order.account_number + "_" + str(order.amount))
	order_info, _ =  sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		trade_logger.info("Close Model Order: " + str({'model_key' : order.model_key, "Base Profit" : str(profit), "Actual Profit" : str(order.PnL)})) 
		return order_metadata, existing_order_amount, True

	return order_metadata, existing_order_amount, False

def create_order(pair, curr_price, account_numbers, trade_dir, order_amount, existing_order_amount, base_model_key):

	if trade_dir == True:
		account_number = account_numbers[0]
	else:
		account_number = account_numbers[1]

	upper_inc = 0
	lower_inc = 0
	new_order_amount = order_amount
	fok_key = pair + "_" + account_number + "_" + str(order_amount)
	while fok_key in existing_order_amount:
		
		if rand.random() > 0.5 or lower_inc + 1 >= order_amount:
			upper_inc += 1
			new_order_amount = order_amount + upper_inc
			fok_key = pair + "_" + account_number + "_" + str(new_order_amount)
		else:
			lower_inc += 1
			new_order_amount = order_amount - lower_inc
			fok_key = pair + "_" + account_number + "_" + str(new_order_amount)

	existing_order_amount.add(fok_key)
	order_amount = new_order_amount

	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01

	precision = '%.4f'
	if pair[4:] == 'JPY':
		precision = '%.3f'

	if trade_dir == True:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (20000 * pip_size))) + '"}'
		#sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		#sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (20000 * pip_size))) + '"}'
		order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		

	return order_info, account_number, order_amount

def update_orders(orders, group_metadata, order_metadata):

	sequence_id = 0
	for order in orders:

		order.sequence_id = sequence_id
		sequence_id += 1

		order_metadata[str(order.order_id) + "_" + order.account_number] = {
			"model_key" : order.model_key,
			"base_amount" : order.base_amount,
			"sequence_id" : order.sequence_id,
			"account_number" : order.account_number,
			"open_prediction" : order.open_prediction,
			"curr_prediction" : order.curr_prediction,
			"prediction_key" : order.prediction_key,
			"portfolio_wt" : order.portfolio_wt,
			"portfolio_dir" : order.portfolio_dir,
			"amount" : order.amount,
			"dir" : order.dir,
		}

def serialize_order(order):

	return {
			"model_key" : order.model_key,
			"base_amount" : order.base_amount,
			"sequence_id" : order.sequence_id,
			"account_number" : order.account_number,
			"open_prediction" : order.open_prediction,
			"curr_prediction" : order.curr_prediction,
			"prediction_key" : order.prediction_key,
			"portfolio_wt" : order.portfolio_wt,
			"portfolio_dir" : order.portfolio_dir,
			"amount" : order.amount,
			"dir" : order.dir,
		}

def close_group(group_metadata, balance, float_profit, margin_used, margin_available, orders):

	if 'equity_curve' in group_metadata:
		equity_curve = group_metadata['equity_curve']
	else:
		equity_curve = [balance]

	if 'group_processed_time' in group_metadata:
		last_time = group_metadata['group_processed_time']
	else:
		last_time = 0

	if (time.time() - last_time > 12 * 60 * 60) or (margin_used / margin_available > 1.1):

		equity_mean = np.mean(equity_curve)

		trade_logger.info('Processing Group: Equity Ratio' + str(balance / equity_mean) + \
			', Float Ratio: ' + str(float_profit / total_exposure) + \
			', Margin Ratio: ' + str(margin_used / margin_available)) 

		if balance / equity_mean > 1.1 or float_profit / margin_used > 1.1 or (margin_used / margin_available > 1.1):
			trade_logger.info("Closing Group")
			for order in orders:
				sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

			os.remove(root_dir + "group_metadata_scalper")
			os.remove(root_dir + "order_metadata_scalper")
			os.remove(root_dir + "model_wt")
			return

		group_metadata['equity_curve'].append(balance)
		group_metadata['group_processed_time'] = time.time()
		pickle.dump(group_metadata, open(root_dir + "group_metadata_scalper", 'wb'))

def close_group_trades(pair, orders_by_model, model_key, order_metadata, group_metadata, total_margin_available, total_margin_used, existing_order_amount, curr_price, avg_prices, avg_spreads, total_models, white_base_model_key_set):

	orders = orders_by_model[model_key]

	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01

	commission = avg_spreads[pair] * pip_size

	buy_profit = 0
	sell_profit = 0

	buy_margin_used = 0
	sell_margin_used = 0

	total_pnl = 0
	for order in orders:
		total_pnl += order.PnL

		if order.portfolio_dir:
			buy_profit += order.PnL

			if order.PnL > 0:
				buy_margin_used += order.margin_used
		else:
			sell_profit += order.PnL

			if order.PnL > 0:
				sell_margin_used += order.margin_used

	if model_key + "_min_profit" not in group_metadata:
		group_metadata[model_key + "_min_profit"] = total_pnl

	group_metadata[model_key + "_min_profit"] = min(total_pnl, group_metadata[model_key + "_min_profit"])

	if (model_key + "_last_profit_udpate") not in group_metadata:
		group_metadata[model_key + "_last_profit_udpate"] = time.time()
		pickle.dump(group_metadata, open(root_dir + "group_metadata_scalper", 'wb'))

	hours_since_check = abs(group_metadata[model_key + "_last_profit_udpate"] - time.time())
	hours_since_check /= 60 * 60

	trade_logger.info('Model: ' + model_key + \
			", Float Profit: " + str(total_pnl) + \
			", Min Profit: " + str(group_metadata[model_key + "_min_profit"]) + \
			", Hours Since: " + str(hours_since_check) + \
			", Total Orders: " + str(len(orders)))



	new_orders = []
	for order in orders:

		margin_used_factor = order.margin_used / (order.PnL + (total_margin_used / total_models))

		if order.dir:
			price = avg_prices[order.pair + "_bid"]
		else:
			price = avg_prices[order.pair + "_ask"]

		pnl_factor = order.PnL / ((price - order.open_price) * order.amount)

		trade_logger.info(
			"    Pair: " + str(order.pair) + \
			", Float Profit: " + str(order.PnL) + \
			", Amount: " + str(order.amount) + \
			", pnlfactor: " + str(pnl_factor) + \
			", Portfolio Wt: " + str('%0.2f' % order.portfolio_wt) + \
			", Dir: " + str(order.dir) + \
			", Open Predict: " + str('%0.2f' % order.open_prediction) + \
			", Time Diff: " + str('%0.2f' % order.time_diff_hours) + \
			", Delta: " + str('%0.2f' % (order.open_prediction - order.curr_prediction)))

		if abs(group_metadata[model_key + "_last_profit_udpate"] - time.time()) < 60 * 60 * 12:
			continue

		if (order.dir == True) == (curr_price > order.open_price):
			pip_diff = ((abs(curr_price - order.open_price)) / pip_size) 
		else:
			pip_diff = ((-abs(curr_price - order.open_price)) / pip_size)


		if (order.model_key not in white_base_model_key_set) and group_metadata[order.prediction_key + "_curr_spread"] < 5.0: 
			trade_logger.info('Model Not WhiteListed: ' + model_key)
			order_metadata, existing_order_amount, is_success = close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size)
			if is_success == True:
				continue
			
		
		if order.dir == True and buy_profit > buy_margin_used and group_metadata[order.prediction_key + "_curr_spread"] < 5.0: 
			order_metadata, existing_order_amount, is_success = close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size)
			if is_success == True:
				continue
		
		if order.dir == False and sell_profit > sell_margin_used and group_metadata[order.prediction_key + "_curr_spread"] < 5.0: 
			order_metadata, existing_order_amount, is_success = close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size)
			if is_success == True:
				continue

		if abs(order.curr_prediction) > 3 and group_metadata[order.prediction_key + "_curr_spread"] < 5.0: 
			order_metadata, existing_order_amount, is_success = close_order(order, order_metadata, existing_order_amount, curr_price, commission, pip_size)
			if is_success == True:
				continue

		new_orders.append(order)

	if abs(group_metadata[model_key + "_last_profit_udpate"] - time.time()) > 60 * 60 * 12 and group_metadata[order.prediction_key + "_curr_spread"] < 5.0:
		group_metadata[model_key + "_last_profit_udpate"] = time.time()

	if len(new_orders) != len(orders):
		update_orders(new_orders, group_metadata, order_metadata)



def enter_group_trades(pair, white_model, order_metadata, group_metadata, orders_by_model, curr_model_wt, prev_model_wt, free_margin, used_margin, curr_price, account_numbers, existing_order_amount, curr_spread, avg_prices):

	model_key = white_model['model_key'] 
	prediction_key = white_model['prediction_key'] 
	martingale_type = white_model['martingale_type'] 

	prediction = -white_model['prediction'] * white_model['trade_dir']
	base_model_key = model_key + "_" + str(white_model['entry_bias']) + "_" + str(white_model['trade_dir']) + "_" + str(martingale_type)
	
	orders = []
	buy_profit = 0
	sell_profit = 0

	buy_margin_used = 0
	sell_margin_used = 0

	buy_order_num = 0
	sell_order_num = 0
	max_buy_profit = -9999999999
	max_sell_profit = -9999999999

	min_time_diff_buy = 999999999
	min_time_diff_sell = 999999999
	if base_model_key in orders_by_model:

		orders = orders_by_model[base_model_key]

		for order in orders:
			buy_amount += order.base_amount
			sell_amount += order.base_amount
			order.curr_prediction = prediction

			if order.portfolio_dir == True:
				buy_profit += order.PnL
				max_buy_profit = max(max_buy_profit, order.PnL)
				buy_order_num += 1

				min_time_diff_buy = min(min_time_diff_buy, order.time_diff_hours)
			else:
				sell_profit += order.PnL
				max_sell_profit = max(max_sell_profit, order.PnL)
				sell_order_num += 1

				min_time_diff_sell = min(min_time_diff_sell, order.time_diff_hours)

		update_orders(orders, group_metadata, order_metadata)
	
		prev_dir = group_metadata[base_model_key + "_prev_dir"]
		if (prediction < 0) != prev_dir:
			group_metadata[base_model_key + "_reduce_amount"] = white_model['entry_bias']
			prev_dir = (prediction < 0)
			group_metadata[base_model_key + "_prev_dir"] = prev_dir
			pickle.dump(group_metadata, open(root_dir + "group_metadata_scalper", 'wb'))
			trade_logger.info('Reset Model: ' + base_model_key + " " + str(prev_dir)) 
		
	else:
		prev_model_wt[base_model_key] = curr_model_wt

		if base_model_key + "_min_profit" in group_metadata:
			del group_metadata[base_model_key + "_min_profit"]

		if base_model_key + "_last_profit_udpate" in group_metadata:
			del group_metadata[base_model_key + "_last_profit_udpate"]

		if base_model_key in prev_model_wt:
			del prev_model_wt[base_model_key]

	for trade_dir in [True, False]:

		all_JPY = True
		for pair in white_model['wts']:
			if pair[4:7] != 'JPY':
				all_JPY = False


		if trade_dir == True and (max_buy_profit > 0 or min_time_diff_buy < 12):
			continue

		if trade_dir == False and (max_sell_profit > 0 or min_time_diff_sell < 12):
			continue

		for pair in white_model['wts']:

			first_currency = pair[0:3]
			second_currency = pair[4:7]
			wt = white_model['wts'][pair]

			if second_currency != "USD":
				pair_mult = avg_prices[second_currency + "_USD"]
			else:
				pair_mult = 1.0

			new_order = Order()

			if trade_dir == True:
				new_order.base_amount = abs(buy_profit) + 1
			else:
				new_order.base_amount = abs(sell_profit) + 1

			if base_model_key not in prev_model_wt:
				prev_model_wt[base_model_key] = curr_model_wt

			new_order.base_amount *= curr_model_wt * abs(white_model['wts'][pair]) 

			new_order.dir = (trade_dir == (white_model['wts'][pair] > 0))
			new_order.portfolio_dir = trade_dir

			if all_JPY == True:
				order_amount = int(round((new_order.base_amount) / (100 * pair_mult)))
			else:
				order_amount = int(round((new_order.base_amount) / (pair_mult)))

			if order_amount == 0:
				continue

			order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount, existing_order_amount, base_model_key)

			print str(order_info)
			order_info = json.loads(order_info)

			if 'orderFillTransaction' in order_info:
				trade_logger.info('New Order: ' + str(order_info)) 
				order_id = str(order_info['orderFillTransaction']['id'])
				group_metadata[base_model_key + "_last_profit_udpate"] = time.time()

				group_metadata[base_model_key + "_prev_dir"] = (prediction < 0)


				new_order.order_id = order_id
				new_order.amount = order_amount
				new_order.account_number = account_number
				new_order.model_key = base_model_key
				new_order.open_prediction = prediction
				new_order.curr_prediction = prediction
				new_order.prediction_key = prediction_key
				new_order.portfolio_wt = white_model['wts'][pair]
				orders.append(new_order)

				trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 

				update_orders(orders, group_metadata, order_metadata)

			else:
				trade_logger.info('Order Error: ' + str(order_info)) 

	return base_model_key


def process_pending_trades(account_numbers, avg_spreads):

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	for account_number in account_numbers:
		response_value, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/summary", "GET")
		j = json.loads(response_value)

		account_profit = float(j['account'][u'unrealizedPL'])
		account_balance = float(j['account'][u'balance'])
		margin_available = float(j['account']['marginAvailable'])
		margin_used = float(j['account']['marginUsed'])

		total_balance += account_balance
		total_float_profit += account_profit
		total_margin_available += margin_available
		total_margin_used += margin_used

	trade_logger.info('Equity: ' + str(total_balance + total_float_profit))

	whitelist = pickle.load(open("/tmp/model_pairs_predictions", 'rb'))

	white_base_model_key_set = set()
	white_list = pickle.load(open(root_dir + "/trading_data/scalper_model_whitelist", 'rb'))

	for white_model in white_list:
		model_key = white_model['model_key'] 
		martingale_type = white_model['martingale_type'] 

		base_model_key = model_key + "_" + str(white_model['entry_bias']) + "_" + str(white_model['trade_dir']) + "_" + str(martingale_type)
		white_base_model_key_set.add(base_model_key)

	if os.path.isfile(root_dir + "group_metadata_scalper"):
		group_metadata = pickle.load(open(root_dir + "group_metadata_scalper", 'rb'))
	else:
		group_metadata = {}

	if os.path.isfile(root_dir + "order_metadata_scalper"):
		order_metadata = pickle.load(open(root_dir + "order_metadata_scalper", 'rb'))
	else:
		order_metadata = {}

	if os.path.isfile(root_dir + "group_base_wt_scalper"):
		prev_model_wt = pickle.load(open(root_dir + "group_base_wt_scalper", 'rb'))
	else:
		prev_model_wt = {}

	if "max_equity" not in group_metadata:
		group_metadata["max_equity"] = total_balance

	group_metadata["max_equity"] = max(group_metadata["max_equity"], total_balance)

	orders = []
	total_margin = 0
	for account_number in account_numbers:
		orders1, total_margin = get_open_trades(account_number, order_metadata, total_margin)
		orders += orders1

	orders_by_prediction = {}
	orders_by_model = {}
	time_diff_hours = {}
	existing_order_amount = set()

	for order in orders:

		if order.prediction_key not in orders_by_prediction:
			orders_by_prediction[order.prediction_key] = []

		if order.model_key not in orders_by_model:
			time_diff_hours[order.model_key] = 9999999999
			orders_by_model[order.model_key] = []

		if order.pair + "_" + order.account_number + "_" + str(order.amount) in existing_order_amount:
			trade_logger.info("Duplicate FIFO trade")

		existing_order_amount.add(order.pair + "_" + order.account_number + "_" + str(order.amount))
		orders_by_prediction[order.prediction_key].append(order)
		orders_by_model[order.model_key].append(order)

		time_diff_hours[order.model_key] = min(time_diff_hours[order.model_key], order.time_diff_hours)

	total_orders = len(orders)

	avg_prices = {}
	pair_bid_ask_map = {}
	for pair in currency_pairs:
		response, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v1/prices?instruments=" + pair, "GET")
		response = json.loads(response)['prices']

		pip_size = 0.0001
		if pair[4:] == "JPY":
			pip_size = 0.01

		bid = None
		ask = None
		for spread_count in range(1):
			curr_spread = 0
			for price in response:
				if price['instrument'] == pair:
					curr_price = (price['bid'] + price['ask']) / 2
					curr_spread = abs(price['bid'] - price['ask']) / pip_size
					bid = price['bid']
					ask = price['ask']
					break

			if curr_price == 0:
				print "price not found"
				continue

		pair_bid_ask_map[pair] = {}
		pair_bid_ask_map[pair]['bid'] = bid
		pair_bid_ask_map[pair]['ask'] = ask

		avg_prices[pair + "_bid"] = bid
		avg_prices[pair + "_ask"] = ask

		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[first_currency + "_" + second_currency] = curr_price

	for pair in currency_pairs:
		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[second_currency + "_" + first_currency] = 1.0 / avg_prices[pair]



	white_base_model_key_set = set()
	scalper_list = pickle.load(open(root_dir + "/trading_data/scalper_model_whitelist", 'rb'))

	scalper_list = scalper_list[:10]

	for white_model in scalper_list:
		model_key = white_model['model_key'] 
		prediction_key = white_model['prediction_key'] 
		martingale_type = white_model['martingale_type'] 
		wts = white_model['wts']

		base_model_key = model_key + "_" + str(white_model['entry_bias']) + "_" + str(white_model['trade_dir']) + "_" + str(martingale_type)
		white_base_model_key_set.add(base_model_key)

		pairs = white_model['currency_pairs']

		curr_spread = 0
		curr_price = 0
		for pair in wts:

			ideal_spread = avg_spreads[pair]
			pip_size = 0.0001
			if pair[4:] == "JPY":
				pip_size = 0.01

			actual_spread = abs(pair_bid_ask_map[pair]['bid'] - pair_bid_ask_map[pair]['ask']) / pip_size
			actual_spread /= ideal_spread 
			curr_price += abs(pair_bid_ask_map[pair]['bid'] + pair_bid_ask_map[pair]['ask']) / pip_size
			curr_spread += actual_spread

		curr_spread /= len(wts)
		curr_price /= len(wts)

		last_processed_key = prediction_key + "_last_processed"
		last_price_key = prediction_key + "_last_price"
		if last_price_key not in group_metadata:
			group_metadata[last_price_key] = curr_price

		price_diff = abs(curr_price - group_metadata[last_price_key]) / 0.0001
		price_diff /= 10
		curr_spread /= max(1, price_diff)

		print prediction_key, "Curr Spread", curr_spread
		
		if curr_spread < 1.2:
			group_metadata[last_processed_key] = time.time()
			group_metadata[last_price_key] = curr_price

		group_metadata[prediction_key + "_curr_spread"] = curr_spread

	count = 0
	for model in whitelist:

		prediction_key = model['prediction_key']
		if prediction_key + "_curr_spread" not in group_metadata:
			continue

		curr_model_wt = float(total_balance) * 0.01
		curr_spread = group_metadata[prediction_key + "_curr_spread"]

		
		base_model_key = enter_group_trades(pair, model, order_metadata, \
			group_metadata, orders_by_model, curr_model_wt, prev_model_wt, \
			total_margin_available, total_margin_used, curr_price, \
			account_numbers, existing_order_amount, curr_spread, avg_prices)
		

	for base_model_key in orders_by_model:
		close_group_trades(pair, orders_by_model, base_model_key, \
		 	order_metadata, group_metadata, total_margin_available, total_margin_used, \
		  	existing_order_amount, curr_price, avg_prices, avg_spreads, \
		  	len(orders_by_model), white_base_model_key_set)

	pickle.dump(prev_model_wt, open(root_dir + "group_base_wt_scalper", 'wb'))
	pickle.dump(group_metadata, open(root_dir + "group_metadata_scalper", 'wb'))
	pickle.dump(order_metadata, open(root_dir + "order_metadata_scalper", 'wb'))
	trade_logger.info('Total Orders: ' + str(total_orders)) 


accounts = [
	["101-011-9454699-002", "101-011-9454699-003"],
	["101-011-9454699-004", "101-011-9454699-005"],
	["101-011-9454699-006", "101-011-9454699-007"]
]


if get_mac() != 154505288144005:
	avg_spreads = pickle.load(open("/root/pair_avg_spread", 'rb'))
	root_dir = "/root/" 
else:
	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	root_dir = "/tmp/" 


	
trade_logger = setup_logger('first_logger', root_dir + "trade_scalper.log")

#process_pending_trades(["101-001-9145068-002", "101-001-9145068-003", "101-001-9145068-004", "101-001-9145068-005"]) #demo 2
process_pending_trades(accounts[1], avg_spreads) #demo
#process_pending_trades(["001-001-1370090-004", "001-001-1370090-003"])



if get_mac() == 154505288144005:

	print "Can't run locally - run only on remote server"

	root_dir = "/root/"
	
	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.put("/tmp/model_pair_wt", "/tmp/model_pair_wt")
	sftp.put("/tmp/pair_avg_spread", root_dir + "pair_avg_spread")
	sftp.put("/tmp/pair_avg_price", root_dir + "pair_avg_price")

	sftp.put("/tmp/pair_avg_spread", root_dir + "group_metadata_scalper")
	sftp.put("/tmp/pair_avg_price", root_dir + "order_metadata_scalper")

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	for pair in currency_pairs:
		sftp.put("/tmp/model_predictions_" + pair, "/tmp/model_predictions_" + pair)


