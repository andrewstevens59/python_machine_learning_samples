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
from model_predict import predict

class MarkovProcess():


	def init(self, root_dir, pair, time_series, prices, labels, price_range, is_use_residual, is_train = True):

		if is_use_residual:
			file_ext = pair + "_use_resid"
		else:
			file_ext = pair

		if os.path.isfile(root_dir + "markov30_process_test_predictions_" + file_ext) == False:

			if is_train == False:
				return None, None

			#self.calendar = pd.DataFrame.from_records(download_calendar(31536000), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])
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

			pickle.dump(predictions, open(root_dir + "markov30_process_test_predictions_" + file_ext, 'wb'))
			pickle.dump(current_prices, open(root_dir + "markov30_process_test_prices_" + file_ext, 'wb'))

		predictions = pickle.load(open(root_dir + "markov30_process_test_predictions_" + file_ext, 'rb'))
		current_prices = pickle.load(open(root_dir + "markov30_process_test_prices_" + file_ext, 'rb'))


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

	def create_matrix(self, prices, index, time_span):

		count = 5

		for i in range(4):
			try:
				predictions = pd.qcut(prices[max(0, index - time_span):index + 1], count, labels=False)
				break
			except:
				count -= 1

		if count == 1:
			predictions = [0] * len(prices[max(0, index - time_span):index + 1])
			
		prev_label = None

		pred_set = []
		for label in predictions:
			if label != prev_label:
				pred_set.append(label)
				prev_label = label


		matrix = [[0] * 5] * 5
		matrix_count = [[0] * 5] * 5
		for i in range(len(pred_set)):

			found_set = set()
			predict_i = pred_set[i]
			for j in range(i + 1, len(pred_set)):

				predict_j = pred_set[j]

				if math.isnan(predict_i) or math.isnan(predict_j):
					continue

				if predict_j in found_set:
					continue

				found_set.add(predict_j)

				matrix[predict_i][predict_j] += abs(i - j) * (index + 1)
				matrix_count[predict_i][predict_j] += (index + 1)

				if len(found_set) >= 5:
					break
		
		feature_vector = []
		for i in range(5):

			for j in range(5):	

				if matrix_count[i][j] == 0:
					feature_vector.append(-1)
				else:
					feature_vector.append(float(matrix[i][j]) / matrix_count[i][j])

		return feature_vector


	def create_training_set(self, pair, time_series, prices, labels, price_range):

		x = []

		prices = [v[0] for v in prices]
		for index in range(len(prices)):
			
			feature_vector = self.create_matrix(prices, index, 250)
			x.append(feature_vector)

		return x

