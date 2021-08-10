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
from sklearn.metrics import accuracy_score

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm



def get_time_series(symbol, time):
	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=H1&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()

	import json
	j = json.loads(response_value)['candles']

	rates = []
	prices = []
	labels = []

	for index in range(len(j) - 1):
		item = j[index]

		rates.append([item['highMid'] - item['lowMid'], item['closeMid'] - item['openMid']])
		prices.append([item['closeMid']])

		if index < len(j) - 48:
			labels.append(j[index + 47]['closeMid'] - j[index + 1]['openMid'])

	return rates, prices, labels


currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY"
]

def find_closest_neighbour(clusters, point):

	min_dist = 9999999
	cluster_id = None
	for index in range(len(clusters)):

		cluster = clusters[index]

		dist = 0
		for dim in range(len(point)):
 			dist += (point[dim] - cluster[dim]) * (point[dim] - cluster[dim])

		if dist < min_dist:
			min_dist = dist
			cluster_id = index

	return cluster_id

def cluster_price_series(labels, price_series):

	neg_price_mov = []
	pos_price_mov = []

	for index in range(len(labels)):

		if abs(labels[index]) < 1:
			continue

		if labels[index] > 0:
			pos_price_mov.append(price_series[index])
		else:
			neg_price_mov.append(price_series[index])


	cluster_num = 50
	mean_center = []
	kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit(pos_price_mov)

	mean_center += kmeans.cluster_centers_.tolist()


	kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit(neg_price_mov)

	mean_center += kmeans.cluster_centers_.tolist() 

	mean_center = sorted(mean_center, key=lambda x: float(sum(x)) / len(x), reverse=False)

	true_instances = []
	false_instances = []


	x = []
	y = []
	for index in range(len(labels)):

		if abs(labels[index]) < 1:
			continue

		cluster_id = find_closest_neighbour(mean_center, prices[index])
		row = [0] * len(mean_center)
		row[cluster_id] = 1
		x.append(row)
		
		if labels[index] > 0:
			true_instances.append(cluster_id)
			y.append(True)
		else:
			false_instances.append(cluster_id)
			y.append(False)


	
	plt.hist(true_instances, alpha=0.5, bins=range(len(mean_center)))  # arguments are passed to np.histogram
	plt.hist(false_instances, alpha=0.5, bins=range(len(mean_center)))  # arguments are passed to np.histogram
	plt.title("Histogram with 'auto' bins")
	plt.show()
	

	new_price_series = []
	for index in range(len(price_series)):
		cluster_id = find_closest_neighbour(mean_center, price_series[index])
		new_price_series.append(mean_center[cluster_id])

	return new_price_series, mean_center, x, y

global_pip_avg = []
for pair in currency_pairs:

	print pair
	time_series, prices, labels = get_time_series(pair, 5000)

	std = np.std(labels)
	labels = [v / std for v in labels]

	test_split = int(0.8 * len(labels))


	x_train = prices[:test_split]
	y_train = labels[:test_split]

	x_test = prices[test_split:]
	y_test = labels[test_split:]

	price_series, mean_centers, x, y = cluster_price_series(y_train, x_train)

	price_series, mean_centers, x, y = cluster_price_series(y_test, x_test)

	sys.exit(0)


	test_split = int(0.8 * len(x))

	x_train = x[:test_split]
	y_train = y[:test_split]

	x_test = x[test_split:]
	y_test = y[test_split:]

	boosting = GradientBoostingClassifier(random_state=42)
	boosting.fit(x_train, y_train)


	predictions = boosting.predict(x_test)

	print accuracy_score(predictions, y_test)

	break

	






