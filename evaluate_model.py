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


def evaluate(x, y, global_pip_avg, pair, result_map):

	if pair not in result_map:
		result_map[pair] = []

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))

	boosting = GradientBoostingRegressor(random_state=42)
	boosting.fit(x[:len(y)], y)

	fpredictions = boosting.predict(x)
	std = np.std(fpredictions)
	fpredictions = [predict / std for predict in fpredictions]
	next_day_predict = fpredictions[-1]
	test_clf = None

	x = x[:len(y)]

	pip_avg = []
	all_pip_avg = []
	returns = []
	curr_return = 0
	for i in range(4, 5):

		offset_start = int((10.0 / 12) * len(x))
		offset_end = int(1 * len(x))


		x_train = x[0: offset_start] + x[offset_end:]
		y_train = y[0: offset_start] + y[offset_end:]

		x_test = x[offset_start:offset_end] 
		y_test = y[offset_start:offset_end]

		boosting = GradientBoostingRegressor(random_state=42)
		boosting.fit(x_train, y_train)

		test_clf = boosting

		predictions = boosting.predict(x_test)

		predictions = [predict / std for predict in predictions]

		true_count = 0
		false_count = 0

		pip_size = 0.0001

		if pair[4:] == "JPY":
			pip_size = 0.01

		commission = avg_spreads[pair] * pip_size

		for index in range(len(x_test)):

			if (predictions[index] > 0) == (y_test[index] > 0):
				global_pip_avg.append(((abs(y_test[index]) - commission) / pip_size) * abs(predictions[index]))
				all_pip_avg.append(((abs(y_test[index]) - commission) / pip_size) * abs(predictions[index]))

				curr_return += ((abs(y_test[index]) - commission) / pip_size) * abs(predictions[index])
			else:
				global_pip_avg.append(((-abs(y_test[index]) - commission) / pip_size) * abs(predictions[index]))
				all_pip_avg.append(((-abs(y_test[index]) - commission) / pip_size) * abs(predictions[index]))

				curr_return += ((-abs(y_test[index]) - commission) / pip_size) * abs(predictions[index])

			returns.append(curr_return)


			if abs(predictions[index]) < abs(next_day_predict) - 0.25:
				continue

			if abs(predictions[index]) > abs(next_day_predict) + 0.25:
				continue

			if (predictions[index] > 0) == (y_test[index] > 0):
				true_count += 1
				pip_avg.append((abs(y_test[index]) - commission) / pip_size)
			else:
				false_count += 1 
				pip_avg.append((-abs(y_test[index]) - commission) / pip_size)

		break

	all_avg = sum(all_pip_avg) / len(all_pip_avg)
	avg = -1

	if len(pip_avg) > 80:

		avg = sum(pip_avg) / len(pip_avg)

		if avg > 0:
			print "Future Prediction", fpredictions[-1], fpredictions[-2], fpredictions[-3], fpredictions[-4], fpredictions[-5]
			print avg, len(pip_avg)

	if all_avg > 0 or True:
		print "All Avg",  all_avg
		'''
		plt.plot(returns)
		plt.ylabel('some numbers')
		plt.show()
		'''

	trade_dir = "SELL"
	if fpredictions[-1] > 0:
		trade_dir = "BUY"


	result_map[pair].append({"dir" : trade_dir, "all_avg" : all_avg, "pred_avg" : avg, "pred_num" : len(pip_avg), "predictions" : fpredictions[-6:]})

	return global_pip_avg, result_map, test_clf