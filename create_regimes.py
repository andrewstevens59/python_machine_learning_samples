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
from download_calendar import download_calendar
from model_predict import predict

class CreateRegimes():




	def init(self, root_dir, pair, time_series, prices, labels, price_range, is_use_residual, is_train = True):

		#0, 200, 300, 100,  200 in sample

		if is_use_residual:
			file_ext = pair + "_use_resid"
		else:
			file_ext = pair

		if os.path.isfile(root_dir + "regime59_model_test_predictions_" + file_ext) == False:

			if is_train == False:
				return None, None

			#self.calendar = pd.DataFrame.from_records(download_calendar(31536000), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])
			x = self.create_training_set(pair, time_series, prices, labels, price_range, prices)

			start = 0
			end = 700
			predictions = []
			current_prices = []
			while end < len(labels):
				predictions.append(predict(start, end, x, labels, is_use_residual))
				current_prices.append(prices[end - 1])
				print start
				start += 12
				end += 12

			pickle.dump(predictions, open(root_dir + "regime59_model_test_predictions_" + file_ext, 'wb'))
			pickle.dump(current_prices, open(root_dir + "regime59_model_test_prices_" + file_ext, 'wb'))

		predictions = pickle.load(open(root_dir + "regime59_model_test_predictions_" + file_ext, 'rb'))
		current_prices = pickle.load(open(root_dir + "regime59_model_test_prices_" + file_ext, 'rb'))

		return predictions, current_prices

	def back_test_recent(self, pair, time_series, prices, labels, price_range, history_prices):

		x = self.create_training_set(pair, time_series, prices, labels, price_range, prices)

		end = len(labels) - (40 * 24)
		start = end - 700
		predictions = []
		current_prices = []
		while end < len(labels):
			predictions.append(predict(start, end, x, labels))
			current_prices.append(prices[end - 1])
			print start
			start += 12
			end += 12

		return predictions, current_prices

	def make_prediction(self, pair, time_series, prices, labels, price_range, history_prices, is_use_residual):

		train_start = len(labels) - 700
		train_end = len(labels)
		x = self.create_training_set(pair, time_series, prices, labels, price_range, history_prices)

		return predict(train_start, train_end, x, labels, is_use_residual)


	def create_training_set(self, pair, time_series, prices, labels, price_range, history_prices):


		kmeans = KMeans(n_clusters=150, init='k-means++', max_iter=100, n_init=1, 
	                               random_state = 42).fit(history_prices)

		predictions = kmeans.predict(prices)
		mean_center = kmeans.cluster_centers_.tolist()

		mean_center = [v[0] for v in mean_center]
		prices = [p[0] for p in prices]

		net_value_long = [0] * len(mean_center)
		net_value_short = [0] * len(mean_center)

		x = []
		y = []

		regime_size = 0
		for index in range(len(prices)):

			feature_vector = []

			cluster_id = predictions[index]

			net_value_long[cluster_id] += 1
			net_value_short[cluster_id] += 1

			if index >= 24:
				cluster_id = predictions[index - 24]
				net_value_short[cluster_id] -= 1

			for offset in range(len(net_value_long)):

				if net_value_long[offset] < 0:
					print "long offset"
					sys.exit(0)

				if net_value_short[offset] < 0:
					print "short offset"
					sys.exit(0)

			norm =  LA.norm(net_value_long)
			long_state_vector = [float(value) / norm for value in net_value_long]

			norm =  LA.norm(net_value_short)
			short_state_vector = [float(value) / norm for value in net_value_short]

			dist = np.dot(long_state_vector, short_state_vector)

			price_subset = prices[index - regime_size : index + 1]

			model = LinearRegression()
			results1 = model.fit([[v] for v in range(len(price_subset))], price_subset)
			price_mean = np.mean(price_subset)
			price_std = np.std(price_subset)

			regime_size += 1

			sorted_set = []
			for offset in range(len(long_state_vector)):
				sorted_set.append([long_state_vector[offset], mean_center[offset]])

			sorted_set = sorted(sorted_set, key=lambda x: abs(x[0]), reverse=True)
			sorted_set = sorted_set[:15]

			feature_vector += [(prices[index] - item[1]) * abs(item[0]) for item in sorted_set]

			price_subset = prices[max(0, index - 24) : index + 1] + ([prices[0]] * max(0, 24 - index))

			model = LinearRegression()
			results2 = model.fit([[v] for v in range(len(price_subset))], price_subset)

			yhat = model.predict([[v] for v in range(len(price_subset))])

			feature_vector += [a - b for a, b in zip(price_subset, yhat)]
			feature_vector.append(results1.coef_[0])
			feature_vector.append(results2.coef_[0])
			feature_vector.append(results2.coef_[0] - results1.coef_[0])
			
	
			while regime_size > 200:

				prev_time_step = index - regime_size + 1
				cluster_id = predictions[prev_time_step]
				net_value_long[cluster_id] -= 1

				regime_size -= 1

				norm =  LA.norm(net_value_long)
				long_state_vector = [float(value) / norm for value in net_value_long]

				dist = np.dot(long_state_vector, short_state_vector)
				
			x.append(feature_vector)

		return x

