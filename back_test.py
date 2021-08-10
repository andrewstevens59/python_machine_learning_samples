

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

import back_test_strategies as back_test
import delta_process as delta_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import train_and_back_test_all as back_test_all
import logging

import plot_equity as portfolio
from uuid import getnode as get_mac
import random as rand
import socket



def load_time_series(start, end):

	with open('/Users/callummc/Downloads/AUDJPY_Candlestick_1_Hour_BID_03.01.2016-30.06.2018.csv') as f:
	    content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	print content[0]


	rates = []
	prices = []
	labels = []
	price_range = []

	content = content[1:]

	for index in range(len(content)):

		if index < start:
			continue

		if index > end:
			break

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


#currency_pairs = ["CAD_JPY", "CHF_JPY", "GBP_JPY", "NZD_JPY", "AUD_JPY", "USD_JPY", "EUR_JPY"]



currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


equity = None

times = []
global_returns = []
global_sharpes = {}
global_currency_pairs = [] 

model_predictions = []
model_prices = []
model_price_change = []

remove_model_pair = set()


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

if get_mac() == 154505288144005:
	root_dir = "/Users/callummc/Documents/trading_data/"
else:
	root_dir = "/root/"


trade_logger = setup_logger('first_logger', root_dir + "whitelist.log")
entry_biases = [3]
trade_dirs = [1, -1]



def neg_sharpe_ratio(weights, equity_curves):

	last_return = [0] * 100000 
	for equity_curve, weight in zip(equity_curves, weights):

		offset = 0
		while offset < (len(equity_curve) - 1000):
			last_return[offset] += (equity_curve[-(offset + 1)] - equity_curve[-(offset + 1000)]) * weight
			offset += 500

	last_return = [v for v in last_return if abs(v) > 0]

	return -np.mean(last_return) / np.std(last_return)

def max_sharpe_ratio(equity_curves, optimization_bounds = (0.0, 1.0)):

   # constraints = [{'type': 'ineq', 'fun': lambda x: +x[i] - avg_weight * max_exposure} for i in range(num_assets)]
    
    num_assets = len(equity_curves)
    constraints = []

    from scipy.optimize import minimize
    args = (equity_curves)
    bound = optimization_bounds
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    print result
    weights = result['x'].tolist() 
    return weights	


white_list = pickle.load(open(root_dir + "model_whitelist_temp", 'rb'))
white_list = sorted(white_list, key=lambda x: x['avg_profit'] / x['draw_down'], reverse=True)

for item in white_list:
	item['closed_profits'] = list(np.random.choice(item['closed_profits'], size=int(len(item['closed_profits']) * 0.75), replace=False))


cutoff = 0.08
equity_curves = []
final_white_list = []

avg_sharpe = []

for trade_dir in trade_dirs:
	mean_sharpe_by_model = {}
	mean_sharpe_by_pair = {}
	all_closed_trades = []

	print "trade_dir", trade_dir
	for item in white_list:
		if item['trade_dir'] != trade_dir:
			continue

		all_closed_trades += item['closed_profits']
		if item['num_trades'] > 1:

			if item['currency_pairs'][0] not in mean_sharpe_by_pair:
				mean_sharpe_by_pair[item['currency_pairs'][0]] = []

			mean_sharpe_by_pair[item['currency_pairs'][0]] += item['closed_profits']

	for item in white_list:
		if item['trade_dir'] != trade_dir:
			continue

		if item['num_trades'] > 1:
			if item['model_key'][7:] not in mean_sharpe_by_model:
				mean_sharpe_by_model[item['model_key'][7:]] = []

			if np.mean(mean_sharpe_by_pair[item['currency_pairs'][0]]) > 0:
				mean_sharpe_by_model[item['model_key'][7:]] += item['closed_profits']

	print "Mean Sharpe", np.mean(all_closed_trades)

	final_sharpe_by_model = {}
	final_sharpe_by_pair = {}
	for model_key in mean_sharpe_by_model:
		print "Model Sharpe", model_key, np.mean(mean_sharpe_by_model[model_key]) / np.std(mean_sharpe_by_model[model_key]), len(mean_sharpe_by_model[model_key])
		final_sharpe_by_model[model_key] = np.mean(mean_sharpe_by_model[model_key]) / np.std(mean_sharpe_by_model[model_key])

	for pair in mean_sharpe_by_pair:
		print "Pair Sharpe", pair, np.mean(mean_sharpe_by_pair[pair]) / np.std(mean_sharpe_by_pair[pair]), len(mean_sharpe_by_pair[pair])
		final_sharpe_by_pair[pair] = np.mean(mean_sharpe_by_pair[pair]) / np.std(mean_sharpe_by_pair[pair])

	for item in white_list:
		if item['trade_dir'] != trade_dir:
			continue

		if item['num_trades'] > 1 and final_sharpe_by_model[item['model_key'][7:]] > 0.03 and final_sharpe_by_pair[item['currency_pairs'][0]] > 0:
			#item['equity_curve'] = []
			final_white_list.append(item)
			equity_curves.append(item['equity_curve'])

print "Total Num", len(final_white_list)

pickle.dump(final_white_list, open(root_dir + "basket_model_whitelist", 'wb'))
print "Saved"

#weights = max_sharpe_ratio(equity_curves, optimization_bounds = (1.0, 2.0))

for item in final_white_list:
	item['weight'] = 1.0


text_file = open("/tmp/replay.txt", "w")


count = 0
last_return = [0] * 100000 
for item in final_white_list:
	count += 1

	print item['model_key'], item['sharpe'], item['r2_score']

	item['equity_curve'] = item['equity_curve'][-1000:]

	i = 0
	offset = 0
	while i < len(item['equity_curve']) - (40):
		last_return[offset] += (item['equity_curve'][-(i + 1)] - item['equity_curve'][-(i + (40))]) * item['weight']
		i += (40)
		offset += 1

	item['equity_curve'] = []

	replay_trades = item['replay_trades']
	text_file.write(str(item) + "\n")

text_file.close()

last_return = [v for v in last_return if abs(v) > 0]
print "last year ", last_return
print "mean", np.mean(last_return) / np.std(last_return)


pickle.dump(final_white_list, open(root_dir + "basket_model_whitelist", 'wb'))

equity_curve = [0]
for index in range(len(last_return)):
	equity_curve.append(equity_curve[-1] + last_return[-(index + 1)])


import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


with PdfPages('/Users/callummc/Desktop/equity_curve.pdf') as pdf:
	plt.figure(figsize=(6, 4))
	plt.plot(equity_curve)
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close() 


	negative_data = []
	negative_index = []
	positive_data = []
	positive_index = []

	for index in range(len(last_return)):
		if last_return[index] > 0:
			positive_data.append(last_return[index])
			positive_index.append(index)
		else:
			negative_data.append(last_return[index])
			negative_index.append(index)


	plt.figure(figsize=(6, 4))
	plt.bar(positive_index, positive_data, width=1, color='b')
	plt.bar(negative_index, negative_data, width=1, color='r')
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close() 

sys.exit(0)



'''
white_list = []
for martingale_type in ["flip", "increase"]:
	for trade_dir in trade_dirs:
		for entry_bias in entry_biases:
			for pair in currency_pairs:

				white_list = back_test_all.train_and_back_test(trade_logger, pair, martingale_type, \
							global_currency_pairs, avg_spreads, avg_prices, entry_bias, trade_dir, white_list)

print "White List Size", len(white_list)
trade_logger.info("White List Size: " + str(len(white_list)))
pickle.dump(white_list, open(root_dir + "model_whitelist", 'wb'))
print "done"

sys.exit(0)
'''

'''
for martingale_type in ["flip", "increase"]:
	for trade_dir in trade_dirs:
		for entry_bias in entry_biases:
			for series_key in stat_arb_series_map.keys()[:24]:

				prices, labels = get_portfolio_series(stat_arb_series_map[series_key])
				print series_key, portfolio_wts[series_key]

				white_list = back_test_all.train_and_back_test_stat_arb(trade_logger, str(series_key), martingale_type, prices, labels, \
							global_currency_pairs, avg_spreads, avg_prices, entry_bias, trade_dir, white_list)
'''

'''
net_returns = None

for pair in currency_pairs:

	print pair

	if pair in ["AUD_JPY", "CAD_JPY", "GBP_JPY", "EUR_JPY", "CHF_JPY", "USD_JPY", "NZD_JPY"]:
		continue

	rates, prices, labels, price_range, times, volumes, returns = back_test_all.get_time_series(pair, 5000)

	if net_returns == None:
		net_returns = [0] * len(returns)

	for index in range(len(returns)):
		net_returns[index] += returns[index]


equity_curve = [0]
for index in range(len(net_returns)):
	equity_curve.append(net_returns[index] + equity_curve[-1])

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


with PdfPages('/Users/callummc/Desktop/equity_curve.pdf') as pdf:

	plt.figure(figsize=(6, 4))
	plt.plot(equity_curve)
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close() 

sys.exit(0)
'''

avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
avg_prices = pickle.load(open("/tmp/pair_avg_price", 'rb'))

'''
portfolio_wts = pickle.load(open(root_dir + "portfolio_wts_min_volatility", 'rb'))

from parallel import *
jobs = []


for martingale_type in ["increase"]:
	for trade_dir in trade_dirs:
		for entry_bias in entry_biases:
			for series_key in portfolio_wts.keys():

				if min(portfolio_wts[series_key]['wt'].values()) < 0.001:
					continue

				jobs.append(
			        WorkItem(
			            fn=back_test_all.train_and_back_test_stat_arb,
			            args=(),
			            kwargs=dict(
			                trade_logger = trade_logger,
			                model_key = str(series_key), 
			                pair = str(series_key), 
			                martingale_type = martingale_type, 
							global_currency_pairs = global_currency_pairs, 
							avg_spreads = avg_spreads, 
							avg_prices = avg_prices, 
							entry_bias = entry_bias, 
							trade_dir = trade_dir,
							is_train_model = True,
							root_dir = root_dir
			            )
			        ) 
			    )

white_list = ProcessPoolExecutor().run(jobs, show_progress_bar = False)

white_list = [item for sublist in white_list for item in sublist]

pickle.dump(white_list, open(root_dir + "model_whitelist_temp", 'wb'))
'''


from parallel import *
jobs = []

for is_use_residual in [False]:
	for martingale_type in ["increase"]:
		for trade_dir in trade_dirs:
			for entry_bias in entry_biases:
				for series_key in currency_pairs:

					jobs.append(
				        WorkItem(
				            fn=back_test_all.train_and_back_test_stat_arb,
				            args=(),
				            kwargs=dict(
				                trade_logger = trade_logger,
				                model_key = str(series_key + "_basket"), 
				                pair = series_key,
				                is_use_residual = is_use_residual,
				                martingale_type = martingale_type, 
								global_currency_pairs = global_currency_pairs, 
								avg_spreads = avg_spreads, 
								avg_prices = avg_prices, 
								entry_bias = entry_bias, 
								trade_dir = trade_dir,
								is_train_model = False,
								root_dir = root_dir
				            )
				        ) 
				    )

white_list = ProcessPoolExecutor().run(jobs, show_progress_bar = False)

white_list = [item for sublist in white_list for item in sublist]

pickle.dump(white_list, open(root_dir + "model_whitelist_temp", 'wb'))

