import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from random import *
import time
import os.path

from sklearn import datasets, linear_model


from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import numpy as np
from sklearn.decomposition import PCA

import pickle


currency_pairs = {
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
}



currencies = ['USD', 'EUR', 'NZD', 'AUD', 'CAD', 'GBP', 'JPY', 'CHF']


def fitSigmaThreshold(sigma_min, is_reverse):

	signal_map = {}
	signal_count_map = {}
	pair_signal_map = {}
	last_trade_time = {}
	currency_exposure = {}
	avg_daily_mov = {}

	pair_active = {}

	win_map = {}
	loss_map = {}

	save_model_dir = "/tmp/EUR_USD_"		
	series_global = pickle.load(open(save_model_dir + "residuals_future0", 'rb'))
	for currency in currencies:
		last_trade_time[currency] = -1000000
		currency_exposure[currency] = 0
		for i in range(len(series_global)):
			signal_map[currency + str(i)] = 0
			signal_count_map[currency + str(i)] = 0

	price_series_map = {}

	for pair in currency_pairs:
		currency1 = pair[0:3]
		currency2 = pair[4:7]
		last_trade_time[pair] = 0

		save_model_dir = "/tmp/" + pair + "_"

		prices = pickle.load(open(save_model_dir + "price_series", 'rb'))

		price_series_map[pair] = prices
		pair_active[pair] = 0

		series1 = {}
		for observed_days in range(8):
			series1[observed_days] = pickle.load(open(save_model_dir + "residuals_future" + str(observed_days), 'rb'))

		for i in range(len(series1[0])):

			avg_pair = 0
			for observed_days in range(8):
				avg_pair = avg_pair + series1[observed_days][i]

			# increment by the average residual for a given pair
			signal_map[currency1 + str(i)] = signal_map[currency1 + str(i)] + (avg_pair / 8)
			signal_map[currency2 + str(i)] = signal_map[currency2 + str(i)] - (avg_pair / 8)

			signal_count_map[currency1 + str(i)] = signal_count_map[currency1 + str(i)] + 1
			signal_count_map[currency2 + str(i)] = signal_count_map[currency2 + str(i)] + 1


	class Order:

	    def __init__(self):
	        self.pair = ""
	        self.dir = 0
	        self.open_price = 0
	        self.time = 0
	        self.amount = 0
	        self.expire_time = 0


	# find the average residual fora given pair
	for i in range(len(series_global)):
		for currency in currencies:
			signal_map[currency + str(i)] = signal_map[currency + str(i)] / signal_count_map[currency + str(i)]


	for currency in currencies:
		print currency, signal_map[currency + str(len(series_global) - 1)] 


	order_queue = []

	profit = 0


	move_amount = {}
	avg_hold_time = {}
	avg_num = {}
	profit_set = {}
	order_amount = {}
	avg_ret_pair = {}

	for pair in currency_pairs:
		avg_hold_time[pair] = 0
		move_amount[pair] = []
		profit_set[pair] = []
		order_amount[pair] = []
		avg_daily_mov[pair] = []

		for i in range(49):
			win_map[pair + str(i)] = 0
			loss_map[pair + str(i)] = 0
			avg_ret_pair[pair + str(i)] = 0


	sl_num = 0
	tp_num = 0
	time_out_num = 0

	max_profit = 0
	order_size = 1


	for i in range(len(series_global)):
		for pair in currency_pairs:

			currency1 = pair[0:3]
			currency2 = pair[4:7]

			last_trade1 = last_trade_time[currency1]
			last_trade2 = last_trade_time[currency2]

			price = price_series_map[pair][i]

			'''
			if i < len(series_global) - 24:
				avg_daily_mov[pair].append(abs(price_series_map[pair][i+24] - price_series_map[pair][i]))
			'''

			if (i - last_trade1) < 2 or (i - last_trade2) < 2:
				continue

			signal1 = signal_map[currency1 + str(i)]
			signal2 = signal_map[currency2 + str(i)]


			if abs(signal1 - signal2) < sigma_min:
				continue

			
			last_trade_time[currency1] = i
			last_trade_time[currency2] = i

			order = Order()
			order.pair = pair
			order.open_price = price

			if is_reverse == True:
				order.dir = signal1 - signal2
			else:
				order.dir = signal2 - signal1

			order.time = i
			order.amount = (abs(signal1) + abs(signal2)) / 2
			order.expire_time = 48
			order_size = order_size + 1


			if order.dir < 0:
				currency_exposure[currency1] = currency_exposure[currency1] + 1
				currency_exposure[currency2] = currency_exposure[currency2] - 1
			else:
				currency_exposure[currency1] = currency_exposure[currency1] - 1
				currency_exposure[currency2] = currency_exposure[currency2] + 1

			order_queue.append(order)

			pair_active[pair] = 1

		new_order_queue = []
		for j in range(len(order_queue)):
			order = order_queue[j]

			delta = price_series_map[order.pair][i] - order.open_price

			if delta != 0:
				if (delta > 0) == (order.dir < 0):
					win_map[order.pair + str(i - order.time)] = win_map[order.pair + str(i - order.time)] + 1
				else:
					loss_map[order.pair + str(i - order.time)] = loss_map[order.pair + str(i - order.time)] + 1

				if order.dir < 0:
					avg_ret_pair[order.pair + str(i - order.time)] = avg_ret_pair[order.pair + str(i - order.time)] + delta
				else:
					avg_ret_pair[order.pair + str(i - order.time)] = avg_ret_pair[order.pair + str(i - order.time)] - delta

			if i - order.time < order.expire_time:
				new_order_queue.append(order)
				continue

			if order.dir < 0:
				profit_set[order.pair].append(delta)
				order_amount[order.pair].append(order.amount)
				profit = profit + delta
			else:
				profit_set[order.pair].append(-delta)
				order_amount[order.pair].append(order.amount)
				profit = profit - delta

			pair_active[order.pair] = 0

		order_queue = new_order_queue

	print "SL Num: ", sl_num, "TP Num:", tp_num, "Time Out: ", time_out_num, "Trade Num: ", len(profit_set), "Profit: ", profit

	avg_win_loss_ratio = 0
	avg_win_loss_num = 0
	for pair in currency_pairs:

		if len(profit_set[pair]) == 0:
			continue

		std_price = np.std(profit_set[pair])

		if std_price == 0:
			continue

		currency1 = pair[0:3]
		currency2 = pair[4:7]

		commission = 0.001
		if currency2 == 'JPY':
			commission  = commission * 100

		win_num = 0
		loss_num = 0
		bin_win = 0
		avg_ret = 0
		for i in range(len(profit_set[pair])):
			profit = profit_set[pair][i]
			amount = order_amount[pair][i]
			avg_ret = avg_ret + ((profit - commission) * amount)
			if profit > 0:
				bin_win = bin_win + 0.75
			else:
				bin_win = bin_win - 1

		avg_ret = avg_ret / len(profit_set[pair])
		bin_win = bin_win / len(profit_set[pair])

		max_ratio = 0
		total_count = 0
		hold_time = 0
		average_return = 0

		total_win = 0
		total_loss = 0
		for i in range(48):

			total_win = total_win + win_map[pair + str(i)]
			total_loss = total_loss + loss_map[pair + str(i)]


		max_ratio = float(total_win) / float(max(1, total_loss))
		if max_ratio > 0:
			print pair, " Sharp Ratio: ", np.mean(profit_set[pair]) / std_price, "Win/Loss: ", max_ratio, total_count, commission, "Hold Time: ", hold_time
			avg_win_loss_ratio = avg_win_loss_ratio + max_ratio
			avg_win_loss_num = avg_win_loss_num + 1

			global_pair_ratio_map[pair + str(sigma_min)] = max_ratio
			global_pair_hold_map[pair + str(sigma_min)] = hold_time
			global_pair_sample_num_map[pair + str(sigma_min)] = total_count

	print "Average WinLossRatio: ", avg_win_loss_ratio / avg_win_loss_num

	pickle.dump(move_amount, open("/tmp/" + "currency_delta", 'wb'))

	#pickle.dump(avg_daily_mov, open("/tmp/" + "avg_currency_daily_mov", 'wb'))


global_pair_ratio_map = {}
global_pair_hold_map = {}
global_pair_sample_num_map = {}

for i in range(15):
	print "----------- Revert ", (0.5 + (i * 0.1)), "-----------"
	fitSigmaThreshold(0.5 + (i * 0.1), True)


pickle.dump(global_pair_ratio_map, open("/tmp/" + "revert_pair_win_loss_ratio_future", 'wb'))


#pickle.dump(global_pair_ratio_map, open("/tmp/" + "revert_pair_win_loss_ratio_past", 'wb'))



global_pair_ratio_map = {}
global_pair_hold_map = {}
global_pair_sample_num_map = {}

for i in range(15):
	print "----------- Follow ", (0.5 + (i * 0.1)), "-----------"
	fitSigmaThreshold(0.5 + (i * 0.1), False)

pickle.dump(global_pair_ratio_map, open("/tmp/" + "follow_pair_win_loss_ratio_future", 'wb'))


#pickle.dump(global_pair_ratio_map, open("/tmp/" + "follow_pair_win_loss_ratio_past", 'wb'))


