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

import numpy as np

import pickle
import matplotlib.pyplot as plt
import pickle
import math
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

from sklearn import mixture
from subspace import Subspace
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm


def get_trailing_stop_size(symbol, time):
	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=D&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()

	import json
	j = json.loads(response_value)['candles']

	price_range = []

	for index in range(len(j) - 1):
		item = j[index]

		price_range.append(abs(item['highMid'] - item['lowMid']))

	avg_range = float(sum(price_range)) / len(price_range)

	return avg_range * 0.5


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]


result_map = {}
for pair in currency_pairs:
	result_map[pair] = []

dirs = ['/tmp/result_map_gradient.pickle', '/tmp/result_map_create_regimes.pickle', '/tmp/result_map_delta_processing.pickle', '/tmp/result_map_jump_process.pickle', '/tmp/result_map_barrier.pickle']

for dir_name in dirs:
	 results = pickle.load( open(dir_name, "rb" ) )

	 for pair in currency_pairs:

	 	result_map[pair].append(results[pair])


for pair in currency_pairs:


	pip_size = 0.0001
	if pair[4:] == "JPY":
		pip_size = 0.01


	print ""
	print pair, (get_trailing_stop_size(pair, 5) / pip_size)

	for result_list, dir_name in zip(result_map[pair], dirs):

		is_set = True
		for result in result_list:

			if result['all_avg'] < -3:
				continue

			if result['pred_avg'] < 0:
				continue

			if result['pred_num'] < 80:
				continue

			if is_set:
				print "----------------------", dir_name
				is_set = False 

			print result


