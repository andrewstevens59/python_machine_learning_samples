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
from sklearn import preprocessing

from sklearn import mixture
from subspace import Subspace
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from model_predict import predict

class Breakout():


	def find_max(self, prices, time_range):


		price_diff = [0] * 4

		for index in range(4, len(prices)):

			series = prices[max(0, index - time_range):index - 2]

			price_diff.append(max(series))

		return price_diff

	def find_min(self, prices, time_range):


		price_diff = [0] * 4

		for index in range(4, len(prices)):

			series = prices[max(0, index - time_range):index - 2]

			price_diff.append(min(series))

		return price_diff

	def init(self, pair, time_series, prices, labels, price_range):

		if os.path.isfile("/tmp/breakout_model_test_predictions_" + pair) == False:
			#self.calendar = pd.DataFrame.from_records(download_calendar(31536000), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])
			prices_single = [p[0] for p in prices]

			min_price_diffs = {}
			max_price_diffs = {}

			self.subspaces = []
			time_range = 4
			for i in range(10):

				min_price_diffs[i] = self.find_min(prices_single, time_range)
				max_price_diffs[i] = self.find_max(prices_single, time_range)

				#subspace = Subspace(min_price_diffs[i][0:train_offset], max_price_diffs[i][0:train_offset], None, labels[0:train_offset], False, 16)

				#self.subspaces.append(subspace)

				time_range *= 2
				if time_range >= 24 * 30:
					break

			x = self.create_training_set(pair, time_series, prices, labels, price_range, min_price_diffs, max_price_diffs)

			start = 0
			end = 700

			predictions = []
			current_prices = []
			while end < len(labels):
				x_train = x[start:end]
				y_train = labels[start:end]
				mean = np.mean(y_train)
				y_train = [v - mean for v in y_train]

				print start

				self.boosting = GradientBoostingRegressor(random_state=42)
				self.boosting.fit(x_train[:-200], y_train[:-200])

				biased_predictions = self.boosting.predict(x_train)

				mean = np.mean(biased_predictions)
				std = np.std(biased_predictions)

				predictions.append((biased_predictions[-1] - mean) / std)
				current_prices.append(prices[end - 1])

				start += 12
				end += 12

			pickle.dump(predictions, open("/tmp/breakout_model_test_predictions_" + pair, 'wb'))
			pickle.dump(current_prices, open("/tmp/breakout_model_test_prices_" + pair, 'wb'))

		predictions = pickle.load(open("/tmp/breakout_model_test_predictions_" + pair, 'rb'))
		current_prices = pickle.load(open("/tmp/breakout_model_test_prices_" + pair, 'rb'))

		return predictions, current_prices


	def create_training_set(self, pair, time_series, prices, labels, price_range, min_price_diffs = None, max_price_diffs = None):


		if min_price_diffs == None:
			prices_single = [p[0] for p in prices]
			min_price_diffs = {}
			max_price_diffs = {}

			time_range = 4
			for i in range(10):

				min_price_diffs[i] = self.find_min(prices_single, time_range)
				max_price_diffs[i] = self.find_max(prices_single, time_range)

				time_range *= 2
				if time_range >= 24 * 30:
					break

		x = []
		for index in range(len(prices)):

			feature_vector = []
			for i in range(len(max_price_diffs)):

				sensor_value = [min_price_diffs[i][index], max_price_diffs[i][index]]
				#subspace = self.subspaces[len(feature_vector)]

				#cluster_id = subspace.find_closest_state(sensor_value)

				#feature_vector.append(subspace.get_class_separation(cluster_id))

				mid = (min_price_diffs[i][index] + max_price_diffs[i][index]) / 2

				feature_vector.append(prices[index][0] - mid)

				#feature_vector.append(max_price_diffs[i][index])

			x.append(feature_vector)


		return x

	def predict(self, x):
		preds1 = self.clf.predict(x)
		preds2 = self.boosting.predict(x)
		confs = self.clf.predict_proba(x) 

		return preds1, preds2, confs





