

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



def collect_avg_pip_size():

	avg_pip_size = {}
	avg_price = {}
	avg_pip_num = {}

	for i in range(10):
		for pair in currency_pairs:

			print pair

			commission = 0.0003
			pip_size = 0.0001

			if pair[4:] == "JPY":
				commission = 0.03
				pip_size = 0.01

			response, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v1/prices?instruments=" + pair, "GET")
			response = json.loads(response)['prices']

			curr_price = 0
			bid = None
			ask = None
			spread = 0
			for price in response:
				if price['instrument'] == pair:
					curr_price = (price['bid'] + price['ask']) / 2
					spread = abs(price['bid'] - price['ask']) / pip_size
					bid = price['bid']
					ask = price['ask']
					break

			if curr_price == 0:
				print "price not found"
				continue

			print spread

			if pair not in avg_pip_size:
				avg_pip_size[pair] = 0
				avg_pip_num[pair] = 0
				avg_price[pair] = 0


			avg_price[pair] += curr_price
			avg_pip_size[pair] += spread
			avg_pip_num[pair] += 1

			avg_price[pair + "_bid"] = bid
			avg_price[pair + "_ask"] = ask

	for pair in currency_pairs:
		avg_pip_size[pair] /= 10
		avg_price[pair] /= 10

		first_currency = pair[0:3]
		second_currency = pair[4:7]

		avg_price[second_currency + "_" + first_currency] = 1.0 / avg_price[pair]


	pickle.dump(avg_price, open("/tmp/pair_avg_price", 'wb'))

	#pickle.dump(avg_pip_size, open("/tmp/pair_avg_spread", 'wb'))

collect_avg_pip_size()