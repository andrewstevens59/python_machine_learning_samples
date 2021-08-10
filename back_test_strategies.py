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


from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import delta_process as delta_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import train_and_back_test_all as back_test_all
from maximize_sharpe import *

import plot_equity as portfolio
from uuid import getnode as get_mac
import socket
import random

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
		self.actual_amount = 0
		self.is_open = False
		self.fifo_offset = 0
		self.last_check_price = 0
		self.traded_amount = 1
		self.exposure_weight = 0

class Model:

	def __init__(self):
		self.start_equity = 0
		self.open_dir = False
		self.inverts = 0
		self.orders = []

class Tuple:

	def __init__(self, bound, pair, trade_side, model_index):
		self.bound = bound
		self.pair = pair
		self.trade_side = trade_side
		self.model_index = model_index


def back_test_strategies(trade_swap, start_equity, chosen_currency, bracket, time_exposure, time_exp_insts, model_orders, strategies, models, all_times, prev_dir_map, reduce_order_map, avg_spreads, avg_prices, model_weights):

	curr_equity = start_equity
	net_float_profit = 0

	returns = []
	model_returns = {}
	equity_curve = []

	strategy_factor = []
	for strategy_index in range(len(strategies)):
		model_returns[strategy_index] = {}
		model_returns[strategy_index]['returns'] = []
		model_returns[strategy_index]['prev_equity'] = curr_equity
		model_returns[strategy_index]['curr_equity'] = curr_equity
		model_returns[strategy_index]['prev_float_profit'] = 0
		model_returns[strategy_index]['curr_float_profit'] = 0

		strategy_factor.append(1.0)

	brackt_size = 40 * 24
	increase_factor = 40
	leverage = 50

	wt = float(curr_equity + net_float_profit) / 5000
	wt *= increase_factor
	portfolio_wts = [wt / 1] * len(strategies)
	prev_portfolio_wts = [wt / 1] * len(strategies)
	free_margin_growth = 1.0
	
	max_equity = 0
	expoures = []
	draw_downs = []
	order_nums = []
	open_models = []
	closed_amounts = []

	fok_orders = set()
	pair_profit = {}
	max_profit = 0
	current_exposure = 0
	prev_model_prediction = {}
	prev_model_price = {}

	last_index = brackt_size * bracket
	start_index = last_index
	for index in range(brackt_size * bracket, len(all_times)):

		curr_time = calendar.timegm(datetime.datetime.strptime(all_times[index], "%Y.%m.%d %H:%M:%S").timetuple())

		'''
		time_epoch_end = calendar.timegm(datetime.datetime.strptime(all_times[index], "%Y.%m.%d %H:%M:%S").timetuple())

		time_epoch_start = calendar.timegm(datetime.datetime.strptime(all_times[last_index], "%Y.%m.%d %H:%M:%S").timetuple())

		if time_epoch_end - time_epoch_start < 12 * 60 * 60:
			continue
		'''

		last_index = index

		prev_equity = curr_equity
		prev_exposure = current_exposure
		prev_float_profit = net_float_profit
		net_float_profit = 0
		currency_exposure = {}
		total_orders = 0
		total_exposure = 0
		pair_num = 0

		model_exposure = {}
		num_active_models = 0

		max_profit = max(max_profit, prev_float_profit)

		max_open_time = 0
		for strategy_index in range(len(strategies)):

			orders = model_orders[strategy_index].orders
			pair = strategies[strategy_index].pair

			time = all_times[index]
			if time not in models[strategy_index]:

				if strategy_index in prev_model_price:
					curr_price = prev_model_price[strategy_index]
					biased_prediction = prev_model_prediction[strategy_index]
				else:
					curr_price = -1
			else:
				curr_price = models[strategy_index][time]['price']
				biased_prediction = models[strategy_index][time]['prediction']
				prev_model_price[strategy_index] = curr_price
				prev_model_prediction[strategy_index] = biased_prediction


			model_exposure[strategy_index] = 0

			if pair not in pair_profit:
				pair_profit[pair] = 0

			total_orders += len(orders)

			first_currency = pair[0:3]
			second_currency = pair[4:7]

			if first_currency != "USD":
				exposure_weight = avg_prices[first_currency + "_USD"]
			else:
				exposure_weight = 1

			if second_currency == "USD":
				avg_prices[first_currency + "_USD"] = curr_price

			if len(orders) > 0:
				num_active_models += 1
   
			for order in orders:
				order.exposure_weight = exposure_weight
				total_exposure += (order.traded_amount * exposure_weight) / leverage
				model_exposure[strategy_index] += (order.traded_amount * 1) / leverage
				max_open_time = max(max_open_time, index - order.open_time)
		
		order_nums.append(total_orders)
		open_models.append(num_active_models)

		time_epoch_end = calendar.timegm(datetime.datetime.strptime(all_times[index], "%Y.%m.%d %H:%M:%S").timetuple())
		time_epoch_start = calendar.timegm(datetime.datetime.strptime(all_times[start_index], "%Y.%m.%d %H:%M:%S").timetuple())
		time_gap_days = (time_epoch_end - time_epoch_start) / (60 * 60 * 24)

		for strategy_index in range(len(strategies)):

			predict_bounds = strategies[strategy_index].bound
			pair = strategies[strategy_index].pair
			model_index = strategies[strategy_index].model_index
			trade_side = strategies[strategy_index].trade_side

			time = all_times[index]
			if time not in models[strategy_index]:
				if strategy_index in prev_model_price:
					curr_price = prev_model_price[strategy_index]
					biased_prediction = prev_model_prediction[strategy_index]
				else:
					curr_price = -1
			else:
				curr_price = models[strategy_index][time]['price']
				biased_prediction = models[strategy_index][time]['prediction']
				prev_model_price[strategy_index] = curr_price
				prev_model_prediction[strategy_index] = biased_prediction

			if curr_price < 0:
				continue

			model_key = strategy_index
			orders = model_orders[model_key].orders

			prev_dir1 = prev_dir_map[model_key]
			reduce_order = reduce_order_map[model_key]

			first_currency = pair[0:3]
			second_currency = pair[4:7]

			if second_currency != chosen_currency and first_currency != chosen_currency:
				continue

			if second_currency != "USD":
				pair_mult = avg_prices[second_currency + "_USD"]
			else:
				pair_mult = 1.0


			pip_size = 0.0001
			if pair[4:] == "JPY":
				pip_size = 0.01

			commission = avg_spreads[pair] * pip_size

			buy_amount = 0
			sell_amount = 0
			float_profit = 0
			avg_hold_time = 0
			total_amount = 0
			actual_float_profit = 0

			min_buy = 9999999
			min_sell = 9999999

			new_orders = []
			for order in orders:

				if (order.dir == True) == (curr_price > order.open_price):
					pip_diff = abs(curr_price - order.open_price) / pip_size
					base_profit = ((abs(curr_price - order.open_price) - commission) / pip_size) * order.amount
					actual_profit = ((abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult
				else:
					pip_diff = -abs(curr_price - order.open_price) / pip_size
					base_profit = ((-abs(curr_price - order.open_price) - commission) / pip_size) * order.amount
					actual_profit = ((-abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult
	
				actual_float_profit += actual_profit
				float_profit += base_profit
				avg_hold_time += (index - order.open_time) 
				total_amount += order.amount

				if order.dir == True:
					buy_amount += order.amount
					min_buy = min(min_buy, order.amount)
				else:
					sell_amount += order.amount
					min_sell = min(min_sell, order.amount)

				new_orders.append(order)

			orders = new_orders
				

			margin_used = (model_exposure[strategy_index]) / (actual_float_profit + ((curr_equity + prev_float_profit) / max(1, num_active_models)))
			total_exposure_ratio = float(total_exposure) / (prev_equity + prev_float_profit)

			
			if ((margin_used > 1) and len(orders) > 1) or margin_used > 2: 
				new_orders = []
				is_found = False
				for order in orders:
					if (order.dir == True) == (curr_price > order.open_price):
						profit = ((abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult
					else:
						profit = ((-abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult

					if margin_used > 2 or (((order.dir == True) == (buy_amount >= sell_amount)) and is_found == False):
						curr_equity += profit
						total_exposure -= (order.traded_amount * order.exposure_weight) / leverage
						pair_profit[pair] += profit

						closed_amounts.append(profit)

						if profit > 0:
							strategy_factor[strategy_index] *= 1.1
						else:
							strategy_factor[strategy_index] *= 0.9

						strategy_factor[strategy_index] = max(strategy_factor[strategy_index], 0.1)
						strategy_factor[strategy_index] = min(strategy_factor[strategy_index], 10)

						if str(order.dir) + "_" + str(order.traded_amount) in fok_orders:
							fok_orders.remove(str(order.dir) + "_" + str(order.traded_amount))

						model_returns[model_key]['curr_equity'] += profit * pip_size
						is_found = True
						continue

					new_orders.append(order)

				orders = new_orders
			
			
			if len(orders) > 0: 
				new_orders = []
				is_found = False
				for order in orders:
					if (order.dir == True) == (curr_price > order.open_price):
						profit = ((abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult
					else:
						profit = ((-abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult

					if prev_exposure > 1 or ((curr_equity + prev_float_profit) / curr_equity > 1.1 and curr_equity + prev_float_profit > start_equity) or (order.dir != (orders[-1].dir)):# or curr_time - order.open_time >= 24 * 60 * 60 * 2:
						curr_equity += profit
						total_exposure -= (order.traded_amount * order.exposure_weight) / 50
						pair_profit[pair] += profit

						closed_amounts.append(profit)

						if profit > 0:
							strategy_factor[strategy_index] *= 1.1
						else:
							strategy_factor[strategy_index] *= 0.9

						strategy_factor[strategy_index] = max(strategy_factor[strategy_index], 0.1)
						strategy_factor[strategy_index] = min(strategy_factor[strategy_index], 10)

						if str(order.dir) + "_" + str(order.traded_amount) in fok_orders:
							fok_orders.remove(str(order.dir) + "_" + str(order.traded_amount))

						model_returns[model_key]['curr_equity'] += profit * pip_size
						is_found = True
						continue

					new_orders.append(order)
				orders = new_orders
			
			
			
			if len(orders) == 0:
				prev_dir1 = None
	

			actual_float_profit = 0
			for order in orders:
				if (order.dir == True) == (curr_price > order.open_price):
					actual_float_profit += ((abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult
				else:
					actual_float_profit += ((-abs(curr_price - order.open_price) - commission)) * order.traded_amount * pair_mult


			if len(orders) > 0:
				avg_hold_time /= len(orders)

			if prev_dir1 != ((biased_prediction * trade_swap) < 0):
				reduce_order = 1
				hold_time = 0

			if total_exposure < prev_equity + prev_float_profit:
				if (abs(biased_prediction) >= 1 and reduce_order < 1.6):

					order = Order()
					if (biased_prediction * trade_swap) < 0:
						order.dir = True
					else:
						order.dir = False
	

					order.open_time = curr_time
					order.open_price = curr_price
					order.amount = 1
					order.invert_num = 1
					order.is_invert = False

					
					order.amount = 1

					
					if len(orders) > 0:
						order.amount = max(len(orders), orders[-1].amount + 1)

					

					prev_portfolio_wts[strategy_index] += 0.1 * (portfolio_wts[strategy_index] - prev_portfolio_wts[strategy_index])

					if second_currency == 'JPY':
						order.traded_amount = round((order.amount * prev_portfolio_wts[strategy_index]) / (100 * pair_mult))
					else:
						order.traded_amount = round((order.amount * prev_portfolio_wts[strategy_index]) / (pair_mult))

					order.traded_amount *= strategy_factor[strategy_index]

					if first_currency != "USD":
						exposure_weight = avg_prices[first_currency + "_USD"]
					else:
						exposure_weight = 1

					order_exposure = (order.traded_amount * exposure_weight) / leverage

					if order.traded_amount > 0 and (order_exposure + total_exposure) < (prev_equity + prev_float_profit):
						orders.append(order)

						prev_dir1 = order.dir
						reduce_order *= 2

						key = str(order.dir) + "_" + str(order.traded_amount)
						up_inc = 0
						down_inc = 0
						original_val = order.traded_amount
						while key in fok_orders:

							if random.random() > 0.5 or down_inc >= original_val - 1:
								up_inc += 1
								order.traded_amount = original_val + up_inc
							else:
								down_inc += 1
								order.traded_amount = original_val - down_inc

							key = str(order.dir) + "_" + str(order.traded_amount)
						
						fok_orders.add(key)
					else:
						pass
						#print "no", (order_exposure + total_exposure) < (prev_equity + prev_float_profit)

			

			m_returns = model_returns[model_key]
			m_returns['curr_float_profit'] = actual_float_profit
			m_returns['returns'].append((m_returns['curr_equity'] + m_returns['curr_float_profit']) - (m_returns['prev_equity'] + m_returns['prev_float_profit']))
			m_returns['prev_float_profit'] = actual_float_profit
			m_returns['prev_equity'] = m_returns['curr_equity']


			model_orders[model_key].orders = orders
			prev_dir_map[model_key] = prev_dir1
			reduce_order_map[model_key] = reduce_order

			net_float_profit += actual_float_profit


		
		current_exposure = float(total_exposure) / (curr_equity + net_float_profit)
		wt = float(curr_equity + net_float_profit) / 5000
		wt *= increase_factor
		#wt /= max(1, num_active_models)

		portfolio_wts = [wt for p in portfolio_wts]

		expoures.append(current_exposure)

		
		if current_exposure > 1:
			print "over exposed", float(curr_equity + net_float_profit) / start_equity

		returns.append((curr_equity + (net_float_profit)) - (prev_equity + (prev_float_profit)))
		#print "Sharpe", (np.mean(returns) / np.std(returns)) * math.sqrt(252 * 24)

		max_equity = max(max_equity, curr_equity + net_float_profit)
		draw_down = 1 - (float(curr_equity + net_float_profit) / max_equity)
		draw_downs.append(draw_down)

		equity_curve.append(curr_equity + (net_float_profit))

	'''
	import matplotlib.pyplot as plt
	plt.plot(range(len(equity_curve)), equity_curve)
	plt.show()
	'''
	return curr_equity + net_float_profit, float(curr_equity + net_float_profit) / start_equity, True, time_gap_days
	

	'''
	pos_num = 0
	neg_num = 0
	for strategy_index in range(len(strategies)): 
		orders = model_orders[strategy_index].orders
		pair = strategies[strategy_index].pair
		curr_price = model_prices[strategy_index][index][0]

		first_currency = pair[0:3]
		second_currency = pair[4:7]

		if second_currency != "USD":
			pair_mult = avg_prices[second_currency + "_USD"]
		else:
			pair_mult = 1.0


		for order in orders:
			if (order.dir == True) == (curr_price > order.open_price):
				pair_profit[pair] += ((abs(curr_price - order.open_price))) * order.traded_amount * pair_mult
			else:
				pair_profit[pair] += ((-abs(curr_price - order.open_price))) * order.traded_amount * pair_mult
	
	for strategy_index in range(len(strategies)):
		pair = strategies[strategy_index].pair
		if pair_profit[pair] > 0:
			pos_num += 1
		else:
			neg_num += 1

	print pos_num, neg_num, "****"
	print "Pair Profit", pair_profit
	'''

	print "Return", (((curr_equity + net_float_profit) / start_equity) - 1) * 100
	print "Equity", curr_equity, "Float", net_float_profit
	print "Sharpe", (np.mean(returns) / np.std(returns)) * math.sqrt(252 * 24)
	print "Mean Exposure", np.mean(expoures) * 100
	print "Max Exposure", np.max(expoures) * 100

	print "Mean Draw Down", np.mean(draw_downs) * 100
	print "Max Draw Down", np.max(draw_downs) * 100

	print "Max Open Orders", np.max(order_nums)
	print "Mean Open Orders", np.mean(order_nums)

	print "Active Models", np.mean(open_models)

	print "Mean Closed", np.mean(closed_amounts), "Closed Num", len(closed_amounts)


	return (np.mean(returns) / np.std(returns)) * math.sqrt(252 * 24), returns, curr_equity + net_float_profit  - 5000
	
