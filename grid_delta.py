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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs

from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from model_predict import predict



class GridDelta():



	def init(self, root_dir, pair, time_series, prices, labels, price_range, is_use_residual, is_train = True):

		if is_use_residual:
			file_ext = pair + "_use_resid"
		else:
			file_ext = pair

		if os.path.isfile(root_dir + "grid_model_test_predictions_" + file_ext) == False:

			if is_train == False:
				return None, None

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

			pickle.dump(predictions, open(root_dir + "grid_model_test_predictions_" + file_ext, 'wb'))
			pickle.dump(current_prices, open(root_dir + "grid_model_test_prices_" + file_ext, 'wb'))

		predictions = pickle.load(open(root_dir + "grid_model_test_predictions_" + file_ext, 'rb'))
		current_prices = pickle.load(open(root_dir + "grid_model_test_prices_" + file_ext, 'rb'))


		return predictions, current_prices

	def back_test_recent(self, pair, time_series, prices, labels, price_range, history_prices):

		x = self.create_training_set(pair, time_series, prices, labels, price_range, history_prices)

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

		cluster_num = 200
		self.kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit(history_prices)

		x = []
		mean_center = self.kmeans.cluster_centers_.tolist()
		state_vector = [0] * len(mean_center)

		state_vector_plus = [0] * len(mean_center)

		state_vector_neg = [0] * len(mean_center)
		price_history = []

		for index in range(len(prices)):

			price = prices[index][0]

			for center, center_index in zip(mean_center, range(len(mean_center))):

				state_vector[center_index] += price - center[0]

				if price - center[0] > 0:
					state_vector_plus[center_index] += 1
				else:
					state_vector_neg[center_index] += 1

			price_history.append(price)

			if len(price_history) > 48:

				price = price_history[0]

				for center, center_index in zip(mean_center, range(len(mean_center))):


					state_vector[center_index] -= price - center[0]

					if price - center[0] > 0:
						state_vector_plus[center_index] -= 1
					else:
						state_vector_neg[center_index] -= 1

				price_history = price_history[1:]

			sorted_set = [[min(state_vector_plus[offset], state_vector_neg[offset]), mean_center[offset][0]] for offset in range(len(mean_center))]

			sorted_set = sorted(sorted_set, key=lambda x: abs(x[0]), reverse=True)
			sorted_set = sorted_set[:60]
			
			z_scores = []
			for item in sorted_set:

				center = item[1]
				state_value = item[0]

				std = 0
				for p in price_history:
					delta = p - center
					std += delta * delta

				std /= len(price_history)
				std = math.sqrt(std)

				delta = price_history[-1] - center

				if std == 0:
					z_scores.append([0, center])
				else:
					z_scores.append([delta / std, center])


			sorted_set = sorted(z_scores, key=lambda x: x[1], reverse=False)

			x.append([item[0] for item in sorted_set])

		return x





