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

class DeltaProcess():

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

		return jumps

	def init(self, pair, time_series, prices, labels, price_range, train_offset):


		price_jumps = {}
		self.kmeans = {}
		for i in range(10):

			cluster_num = (i * 10) + 50
			self.kmeans[i] = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
	                               random_state = 42).fit(prices)

			price_jumps[i] = self.find_jumps(prices, self.kmeans[i])

		self.subspaces = []
		for i in range(len(price_jumps)):

			for j in range(len(price_jumps)):

				if i <= j:
					continue

				subspace = Subspace(price_jumps[i][0:train_offset], price_jumps[j][0:train_offset], None, labels[0:train_offset], False, 16)

				self.subspaces.append(subspace)

		x = self.create_training_set(pair, time_series, prices, labels, price_range, price_jumps)

		x_train = x[:train_offset]
		y_train = labels[:train_offset]

		self.boosting = GradientBoostingRegressor(random_state=42)
		self.boosting.fit(x_train, y_train)

		std = np.std(labels[:train_offset])

		dir_label = []
		for label in labels[:train_offset]:

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


	def create_training_set(self, pair, time_series, prices, labels, price_range, price_jumps = None):

		if price_jumps == None:
			price_jumps = {}
			for i in range(10):

				cluster_num = (i * 10) + 50
				price_jumps[i] = self.find_jumps(prices, self.kmeans[i])

		x = []
		for index in range(len(prices)):


			feature_vector = []
			for i in range(len(price_jumps)):

				for j in range(len(price_jumps)):

					if i <= j:
						continue


					sensor_value = [price_jumps[i][index], price_jumps[j][index]]
					subspace = self.subspaces[len(feature_vector)]

					cluster_id = subspace.find_closest_state(sensor_value)

					feature_vector.append(subspace.get_class_separation(cluster_id))

			x.append(feature_vector)


		y = labels

		return x

	def predict(self, x):
		preds1 = self.clf.predict(x)
		preds2 = self.boosting.predict(x)
		confs = self.clf.predict_proba(x) 

		return preds1, preds2, confs
	



