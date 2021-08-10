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

class Gradient():

	def init(self, root_dir, pair, time_series, prices, labels, price_range, lag, is_use_residual, is_train = True):

		if is_use_residual:
			file_ext = pair + "_use_resid"
		else:
			file_ext = pair

		if os.path.isfile(root_dir + "grad6_model_test_predictions_" + file_ext) == False:

			if is_train == False:
				return None, None

			x = self.create_training_set(pair, time_series, prices, labels, price_range, lag)

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

			pickle.dump(predictions, open(root_dir + "grad6_model_test_predictions_" + file_ext, 'wb'))
			pickle.dump(current_prices, open(root_dir + "grad6_model_test_prices_" + file_ext, 'wb'))

		predictions = pickle.load(open(root_dir + "grad6_model_test_predictions_" + file_ext, 'rb'))
		current_prices = pickle.load(open(root_dir + "grad6_model_test_prices_" + file_ext, 'rb'))


		return predictions, current_prices

	def make_prediction(self, pair, time_series, prices, labels, price_range, lag, history_prices, is_use_residual):

		train_start = len(labels) - 700
		train_end = len(labels)
		x = self.create_training_set(pair, time_series, prices, labels, price_range, lag)

		return predict(train_start, train_end, x, labels, is_use_residual)

	def create_training_set(self, pair, time_series, prices, labels, price_range, lag):


		x = []
		price_diff = [prices[0][0]] * lag
		for index in range(len(prices)):

			feature_vector = []
			for offset in range(lag):
				feature_vector.append(prices[index][0] - price_diff[-offset])

			x.append(feature_vector)
			price_diff.append(prices[index][0])


		return x

	def predict(self, x):
		preds1 = self.clf.predict(x)
		preds2 = self.boosting.predict(x)
		confs = self.clf.predict_proba(x) 

		return preds1, preds2, confs






