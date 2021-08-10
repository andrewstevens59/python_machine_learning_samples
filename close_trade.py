

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
import train_and_back_test as back_test
import plot_equity as portfolio
from maximize_sharpe import *

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

def sendCurlRequest(url, request_type, post_data = None):
	response_buffer = StringIO()
	header_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, url)

	curl.setopt(pycurl.CUSTOMREQUEST, request_type)

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.setopt(curl.HEADERFUNCTION, header_buffer.write)

	#2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8 actual

	#3168adee62ceb8b5f750efcec83c2509-1db7870026de5cccb33b220c100a07ab demo

	#16736a148307bf5d91f9f03dd7c91623-af6a25cf6249c9083f11593ad3899f89 demo2

	print url


	if post_data != None:
		print post_data
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b', 'Content-Type: application/json'])
		curl.setopt(pycurl.POSTFIELDS, post_data)
	else:
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 653affd294dcdcacc39b3bc0a3827417-1949317e089b06e114afbf339ee22b7b'])

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	header_value = header_buffer.getvalue()

	return response_value, header_value

def collect_avg_pip_size():

	avg_pip_size = {}
	avg_pip_num = {}

	for i in range(10):
		for pair in currency_pairs:

			print pair

			commission = 0.0003
			pip_size = 0.0001

			if pair[4:] == "JPY":
				commission = 0.03
				pip_size = 0.01

			response = sendCurlRequest("https://api-fxtrade.oanda.com/v1/prices?instruments=" + pair, "GET")
			response = json.loads(response)['prices']

			curr_price = 0
			spread = 0
			for price in response:
				if price['instrument'] == pair:
					curr_price = (price['bid'] + price['ask']) / 2
					spread = abs(price['bid'] - price['ask']) / pip_size
					break

			if curr_price == 0:
				print "price not found"
				continue

			print spread

			if pair not in avg_pip_size:
				avg_pip_size[pair] = 0
				avg_pip_num[pair] = 0

			avg_pip_size[pair] += spread
			avg_pip_num[pair] += 1

	for pair in currency_pairs:
		avg_pip_size[pair] /= 10

	pickle.dump(avg_pip_size, open("/tmp/pair_avg_spread", 'wb'))



def close_trades(account_number):

	response_value = sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades?count=500", "GET")

	j = json.loads(response_value)
	open_orders = j['trades']

	pair_time_diff = {}
	for trade in open_orders:

		#print trade['openTime']
		s = trade['openTime'].replace(':', "-")
		s = s[0 : s.index('.')]
		order_id = trade['id']
		open_price = float(trade[u'price'])
		pair = trade[u'instrument']
		pip_size = 0.0001

		if pair[4:7] == "JPY":
			pip_size = 0.01

		week_day_start = datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").weekday()

		time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

		s = str(datetime.datetime.utcnow())
		s = s[0 : s.index('.')]

		week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

		time_end = time.time()


		time_diff_hours = time_end - time_start

		time_diff_hours /= 60 * 60

		if week_day_start != 6 and week_day_end < week_day_start:
			time_diff_hours -= 49
		elif week_day_end == 6 and week_day_end > week_day_start:
			time_diff_hours -= 49

		'''
		response = sendCurlRequest("https://api-fxtrade.oanda.com/v1/prices?instruments=" + pair, "GET")
		response = json.loads(response)['prices']


		curr_price = 0
		for price in response:
			if price['instrument'] == pair:
				curr_price = (price['bid'] + price['ask']) / 2
				break

		if curr_price == 0:
			print "price not found"
			continue

		pip_diff = (curr_price - open_price) / pip_size
		'''

		#print trade[u'instrument'], "Since Last Opened: ", time_diff_hours

		if time_diff_hours >= 48:
			print sendCurlRequest("https://api-fxtrade.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
		

		pair = trade['instrument'].replace("/", "_")

		if pair not in pair_time_diff:
			pair_time_diff[pair] = time_diff_hours
		
		pair_time_diff[pair] = min(pair_time_diff[pair], time_diff_hours)

	return pair_time_diff

