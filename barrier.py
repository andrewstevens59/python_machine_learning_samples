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
import pickle
import math
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from model_predict import predict


class Barrier():

	def init(self, pair, time_series, prices, labels, price_range, lag):


	

		if os.path.isfile("/tmp/barrier1_model_test_predictions_" + pair) == False:

			cluster_num = 4
			self.kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
		                               random_state = 42).fit(prices)

		
			x = self.create_training_set(pair, time_series, prices, labels, price_range, lag)

			start = 0
			end = 700
			predictions = []
			current_prices = []
			while end < len(labels):
				predictions.append(predict(start, end, x, labels))
				current_prices.append(prices[end - 1])
				print end
				start += 12
				end += 12

			pickle.dump(predictions, open("/tmp/barrier1_model_test_predictions_" + pair, 'wb'))
			pickle.dump(current_prices, open("/tmp/barrier1_model_test_prices_" + pair, 'wb'))

		predictions = pickle.load(open("/tmp/barrier1_model_test_predictions_" + pair, 'rb'))
		current_prices = pickle.load(open("/tmp/barrier1_model_test_prices_" + pair, 'rb'))


		return predictions, current_prices

	def create_training_set(self, pair, time_series, prices, labels, price_range, lag):


		mean_center = self.kmeans.cluster_centers_.tolist()


		for barrier in mean_center:

			x = []
			barrier_diff = [(prices[0][0] - barrier[0])] * lag
			price_diff = [0] * lag
			for index in range(len(prices)):

				barrier_diff.append(prices[index][0] - barrier[0])

				feature_vector = []
				for offset in range(lag):
					feature_vector.append(barrier_diff[-offset])

				x.append(feature_vector)


			return x
	






