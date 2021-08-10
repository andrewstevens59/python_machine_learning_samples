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

from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from evaluate_model import evaluate
from download_calendar import download_calendar


class Sequence():

	def init(self, pair, time_series, prices, labels, price_range, train_offset, frac_start, frac_end):

		#self.calendar = pd.DataFrame.from_records(download_calendar(31536000), columns=['currency', 'impact', 'actual', 'forecast', 'time', 'region'])
		x = self.create_training_set(pair, time_series, prices, labels, price_range, train_offset)

		x_train = x[frac_start:train_offset]
		y_train = labels[frac_start:train_offset]

		self.boosting = GradientBoostingRegressor(random_state=42)
		self.boosting.fit(x_train, y_train)

		std = np.std(labels[frac_start:train_offset])

		dir_label = []
		for label in labels[frac_start:train_offset]:

			if abs(label / std) < 1:
				dir_label.append("OTHER")
				continue

			if label > 0:
				dir_label.append("True")
			else:
				dir_label.append("False")

		self.clf = GradientBoostingClassifier(random_state=42)
		self.clf.fit(x_train, dir_label)

		return x


	def create_training_set(self, pair, time_series, prices, labels, price_range, train_offset):

		predictions = {}
		prices = [p[0] for p in prices]
		predictions = pd.qcut(prices, 20, labels=False)

		mean_center = np.percentile(prices, [i * 0.01 for i in range(100)])

		emp_map = {}
		key_count = {}

		avg_value = 0
		avg_count = 0

		sequence = []
		prev_cluster_id = None
		for index in range(len(prices)):

			if index >= train_offset:
				break

			label = labels[index]

			cluster_id = predictions[index]

			if cluster_id != prev_cluster_id:
				sequence.append(cluster_id)
				prev_cluster_id = cluster_id

			subsequence = sequence[max(0, index - 10) : index + 1]

			for i in range(1 << len(subsequence)):

				subset = []
				for j in range(len(subsequence)):

					if (i & (1 << j)) > 0:
						subset.append(subsequence[j])

				key = str(subset)

				if key not in emp_map:
					emp_map[key] = 0
					key_count[key] = 0

				emp_map[key] += label
				key_count[key] += 1

				avg_value += label
				avg_count += 1

		chosen_pred = {}
		chosen_id = {}

		for key in key_count:
			if key_count[key] < 3 or (abs(emp_map[key]) / key_count[key]) < (abs(avg_value) / avg_count) * 0.2:
				continue

			chosen_pred[key] = emp_map[key] / key_count[key]
			chosen_id[key] = len(chosen_id)

		x = []
		for index in range(len(prices)):

			cluster_id = predictions[index]

			if cluster_id != prev_cluster_id:
				sequence.append(cluster_id)
				prev_cluster_id = cluster_id

			subsequence = sequence[max(0, index - 10) : index + 1]

			feature_vector = [0] * len(chosen_id)
			for i in range(1 << len(subsequence)):

				subset = []
				for j in range(len(subsequence)):

					if (i & (1 << j)) > 0:
						subset.append(subsequence[j])

				key = str(subset)

				if key in chosen_id:
					pred = chosen_pred[key]
					feature_id = chosen_id[key]
					feature_vector[feature_id] = pred

	
			x.append(feature_vector)

		return x

	def predict(self, x):
		preds1 = self.clf.predict(x)
		preds2 = self.boosting.predict(x)
		confs = self.clf.predict_proba(x) 

		return preds1, preds2, confs

