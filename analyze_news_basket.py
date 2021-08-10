



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
from maximize_sharpe import *



currencies = ["EUR", "USD", "JPY", "CAD", "AUD", "GBP", "NZD", "CHF"]


from datetime import datetime as dt
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

	for index in range(len(j) - 1):
		item = j[index]

		s = item['time'].replace(':', "-")
		s = s[0 : s.index('.')]

		date_obj = dt.strptime(s, "%Y-%m-%dT%H-%M-%S")
		item['time'] = dt.strftime(date_obj, "%Y.%m.%d %H:%M:%S")

		times.append(item['time'])
		prices.append([item['closeMid']])
		volumes.append([item['volume']])

		if index < len(j) - 48:
			rates.append(j[index + 47]['closeMid'] - j[index]['closeMid'])
			labels.append(j[index + 47]['closeMid'] - j[index]['closeMid'])

	return rates, prices, labels, price_range, times, volumes

def load_time_series(symbol):

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
	times = []

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
		times.append(toks[0])

		if index < len(content) - 48:

			toks = content[index + 48].split(',')

			labels.append(float(toks[4]) - c_price)

	return times, labels, prices



def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)

    net_weight = 0 
    for w, r in zip(weights, mean_returns):
        net_weight += abs(w)

    return -1 / p_var

def max_sharpe_ratio(returns, optimization_bounds = (-1.0, 1.0)):

    df = pd.DataFrame(returns)
    mean_returns = df.mean()
    cov_matrix = df.cov()

    num_assets = len(mean_returns)
    avg_weight = 1.0 / num_assets

   # constraints = [{'type': 'ineq', 'fun': lambda x: +x[i] - avg_weight * max_exposure} for i in range(num_assets)]
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum([abs(x1) for x1 in x]) - 1})
        
    from scipy.optimize import minimize
    args = (mean_returns, cov_matrix, 0)
    bound = optimization_bounds
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result['x'].tolist() 
    return weights	

year = 2020

global_currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

if os.path.isfile("/tmp/all_times" + str(year)) == False:

	price_map = {}
	all_times = []
	for pair in global_currency_pairs:

		times, labels, prices = load_time_series(pair)
		all_times += times
		
		for time, price in zip(times, prices):
			price_map[time + "_" + pair] = price


	all_times = list(set(all_times))
	all_times = sorted(all_times)

	pickle.dump(all_times, open("/tmp/all_times" + str(year), 'wb'))
	pickle.dump(price_map, open("/tmp/price_map" + str(year), 'wb'))

all_times = pickle.load(open("/tmp/all_times" + str(year), 'rb'))
price_map = pickle.load(open("/tmp/price_map" + str(year), 'rb'))

series_suffix = "_min_volatility"

def get_calendar_df(pair, year): 

	currencies = [pair[0:3], pair[4:7]]

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

		if toks[2] in currencies:

			est = datetime.datetime.strptime(toks[0] + " " + toks[1], "%b%d.%Y %H:%M%p")
			est = est.replace(tzinfo=from_zone)
			utc = est.astimezone(to_zone)

			time = calendar.timegm(utc.timetuple())

			contents.append([toks[2], time])

	return pd.DataFrame(contents, columns=["currency", "time"])

def create_basket():


	portfolio_wts = {}
	stat_arb_series = {}
	stat_arb_times = {}

	sharpes = []
	est_sharpes = []
	returns = []
	time_exposure = {}
	time_exp_insts = {}

	prev_price_pair = {}
	curr_price_pair = {}
	currency_pairs = global_currency_pairs

	for pair in currency_pairs:
		prev_price_pair[pair] = None
		curr_price_pair[pair] = None

	prices = []
	times = []
	offset = 0
	for time in all_times:

		not_avail = False
		for pair in currency_pairs:

			if time + "_" + pair in price_map:
				curr_price_pair[pair] = price_map[time + "_" + pair][0]

			if curr_price_pair[pair] == None:
				not_avail = True
				continue

		if not_avail:
			continue

		feature_vector = []
		for pair in currency_pairs:
			feature_vector.append(curr_price_pair[pair])

		prices.append(feature_vector)
		times.append(time)

	print len(prices), len(all_times)
	price_diff = []
	for offset in range(1, len(prices)):

		diff = []
		for index in range(len(currency_pairs)):
			diff.append(prices[offset][index] - prices[offset - 1][index])

		price_diff.append(diff)


	wts = max_sharpe_ratio(price_diff)


	equity_curves = []
	titles = []
	for chosen_pair_index in range(len(currency_pairs)):

		portfolio_wts = {}
		stat_arb_series = {}
		actual_price_series = {}
		combo_key = str(currency_pairs[chosen_pair_index]) + "_basket"
		portfolio_wts[combo_key] = {}
		portfolio_wts[combo_key]['wt'] = {}
		portfolio_wts[combo_key]['currency_pairs'] = currency_pairs
		for pair, index in zip(currency_pairs, range(len(currency_pairs))):
			portfolio_wts[combo_key]['wt'][pair] = wts[index]

		#calendar_df = get_calendar_df(currency_pairs[chosen_pair_index], None)

		norm_prices = []
		actual_prices = []
		stat_arb_series = {}

		equity_curve = []

		trade_dir = None
		open_price = None
		equity = 0
		for offset in range(0, len(prices)):
	 
			norm_price = 0
			for index in range(len(currency_pairs)):
				norm_price += prices[offset][index] * wts[index]

			norm_prices.append(norm_price)
			actual_prices.append(prices[offset][chosen_pair_index])

			history_len = 24 * 5

			if len(norm_prices) < history_len:
				continue

			delta = abs(np.corrcoef(norm_prices[-history_len:], actual_prices[-history_len:])[0, 1]) / abs(np.corrcoef(norm_prices[-24:], actual_prices[-24:])[0, 1])
			delta2 = abs(np.corrcoef(norm_prices[-history_len:], actual_prices[-history_len:])[0, 1]) / abs(np.corrcoef(norm_prices[-12:], actual_prices[-12:])[0, 1])
			

			if delta > 3 and delta2 < 2.0 and open_price == None:
				open_price = prices[offset][chosen_pair_index]
				trade_dir = prices[offset][chosen_pair_index] > prices[offset-24][chosen_pair_index]

			if delta < 0.5:
				if open_price != None:

					if trade_dir == (prices[offset][chosen_pair_index] > open_price):
						equity += abs(prices[offset][chosen_pair_index] - open_price)
					else:
						equity -= abs(prices[offset][chosen_pair_index] - open_price)

					equity_curve.append(equity)

				open_price = None

				print currency_pairs[chosen_pair_index], equity

		equity_curves.append(equity_curve)
		titles.append(currency_pairs[chosen_pair_index])

	import datetime
	from matplotlib.backends.backend_pdf import PdfPages
	import matplotlib.pyplot as plt


	with PdfPages('/Users/callummc/Desktop/equity_curve_' + str(year) + '.pdf') as pdf:

		for equity_curve, title in zip(equity_curves, titles):
			plt.figure(figsize=(6, 4))
			plt.plot(equity_curve)
			plt.title(title)

			pdf.savefig()  # saves the current figure into a pdf page
			plt.close() 


from parallel import *
import itertools

create_basket()

'''
jobs = []
for pair_index1 in range(len(global_currency_pairs)):

	pair1 = global_currency_pairs[pair_index1]

	jobs.append(
        WorkItem(
            fn=find_portfolio,
            args=(),
            kwargs=dict(
                currency_pairs = global_currency_pairs,
                combo = [pair1],
                is_triple = False
            )
        ) 
    )

results = ProcessPoolExecutor().run(jobs, show_progress_bar = False)
'''

