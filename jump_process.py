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
from sklearn import preprocessing

from sklearn import mixture
from subspace import Subspace
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from model_predict import predict

class JumpProcess():

	def find_jumps(self, prices, kmeans):

		predictions = kmeans.predict(prices)

		mean_center = kmeans.cluster_centers_.tolist()

		jump_rate = []
		curr_price = None
		prev_price = prices[0][0]
		prev_cluster_id = None
		prev_time_index = 0
		min_distance = None

		price_history = []
		time_history = []
		jump_offset = []
		for index in range(len(prices)):

			price = prices[index]

			jump_offset.append(len(price_history))
			cluster_id = predictions[index]

			if cluster_id != prev_cluster_id:
				price_history.append(prev_price)
				time_history.append(prev_time_index)
				prev_cluster_id = cluster_id
				min_distance = None

			if min_distance == None or abs(mean_center[cluster_id][0] - price[0]) < min_distance:
				min_distance = abs(mean_center[cluster_id][0] - price[0])
				prev_price = price[0]
				prev_time_index = index
				continue

		price_history.append(prev_price)
		time_history.append(prev_time_index)

		jumps = []
		for index in range(len(prices)):


			offset = jump_offset[index]
			offset = min(offset, len(price_history) - 1)

			if offset == 0:
				jumps.append(0)
				continue

			jumps.append(float(price_history[offset] - price_history[offset - 1]) / float(max(1, time_history[offset] - time_history[offset - 1])))

		return preprocessing.scale(jumps)

	def init(self, root_dir, pair, time_series, prices, labels, price_range, is_use_residual, is_train = True):

		if is_use_residual:
			file_ext = pair + "_use_resid"
		else:
			file_ext = pair


		if os.path.isfile(root_dir + "jump_process64_model_test_predictions_" + file_ext) == False:

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

			pickle.dump(predictions, open(root_dir + "jump_process64_model_test_predictions_" + file_ext, 'wb'))
			pickle.dump(current_prices, open(root_dir + "jump_process64_model_test_prices_" + file_ext, 'wb'))

		predictions = pickle.load(open(root_dir + "jump_process64_model_test_predictions_" + file_ext, 'rb'))
		current_prices = pickle.load(open(root_dir + "jump_process64_model_test_prices_" + file_ext, 'rb'))


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

		price_jumps = {}
		self.kmeans = {}
		for i in range(10):

			cluster_num = (i * 10) + 50
			self.kmeans[i] = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
	                               random_state = 42).fit(history_prices)

			price_jumps[i] = self.find_jumps(prices, self.kmeans[i])

		norm_prices = preprocessing.scale(prices)

		x = []
		for index in range(len(prices)):


			feature_vector = []
			for i in range(len(price_jumps)):

				for j in range(len(price_jumps)):

					if i <= j:
						continue

					feature_vector.append(price_jumps[i][index] - price_jumps[j][index])

				feature_vector.append(price_jumps[i][index])
					
			feature_vector.append(norm_prices[index][0])

			x.append(feature_vector)

		return x

	def predict(self, x):
		preds1 = self.clf.predict(x)
		preds2 = self.boosting.predict(x)
		confs = self.clf.predict_proba(x) 

		return preds1, preds2, confs





