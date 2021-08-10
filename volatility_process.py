import sys
import math
from datetime import datetime
from random import *
import os.path


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

class VolatilityProcess():


	def find_volatility(self, prices, time_range):


		avg_sum = 0

		volatility = [0]
		z_scores = [0]

		for index in range(1, len(prices)):

			series = prices[max(0, index - time_range):index]

			mean = np.std(series)
			std = np.std(series)

			if std == 0:
				volatility.append(0)
				z_scores.append(0)
				continue

			z_score = (prices[index][0] - mean) / std

			volatility.append(std)
			z_scores.append(z_score)

		return z_scores, preprocessing.scale(volatility)


	def init(self, root_dir, pair, time_series, prices, labels, price_range, is_use_residual, is_train = True):

		if is_use_residual:
			file_ext = pair + "_use_resid"
		else:
			file_ext = pair

		if os.path.isfile(root_dir + "volatility41_model_test_predictions_" + file_ext) == False:

			if is_train == False:
				return None, None

			x = self.create_training_set(pair, time_series, prices, labels, price_range)

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

			pickle.dump(predictions, open(root_dir + "volatility41_model_test_predictions_" + file_ext, 'wb'))
			pickle.dump(current_prices, open(root_dir + "volatility41_model_test_prices_" + file_ext, 'wb'))

		predictions = pickle.load(open(root_dir + "volatility41_model_test_predictions_" + file_ext, 'rb'))
		current_prices = pickle.load(open(root_dir + "volatility41_model_test_prices_" + file_ext, 'rb'))

		return predictions, current_prices

	def back_test_recent(self, pair, time_series, prices, labels, price_range, history_prices):

		x = self.create_training_set(pair, time_series, prices, labels, price_range)

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
		x = self.create_training_set(pair, time_series, prices, labels, price_range)

		return predict(train_start, train_end, x, labels, is_use_residual)


	def create_training_set(self, pair, time_series, prices, labels, price_range, z_scores = None, volatility = None):

		z_scores = {}
		volatility = {}

		time_range = 24
		for i in range(10):

			z, v = self.find_volatility(prices, time_range)

			z_scores[i] = z
			volatility[i] = v

			time_range += 24

		x = []
		for index in range(len(prices)):

			feature_vector = []
			for i in range(10):

				for j in range(10):

					if i < j:
						continue

					feature_vector.append(z_scores[i][index] - z_scores[j][index])
					
			feature_vector.append(prices[index][0])
			x.append(feature_vector)


		return x






