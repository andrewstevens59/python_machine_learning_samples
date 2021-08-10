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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from model_predict import predict

class Subspace:

	def __init__(self, jumps1, jumps2, prices, labels, is_include_price, cluster_num):


		merged_series = []

		if is_include_price:
			for index in range(len(labels)):
				merged_series.append([prices[index][0], jumps1[index], jumps2[index]])
		else:
			for index in range(len(labels)):
				merged_series.append([jumps1[index], jumps2[index]])


		#self.scaler = preprocessing.StandardScaler().fit(merged_series)
		#merged_series = self.scaler.transform(merged_series)


		sensor_true_df = []
		sensor_false_df = []
		self.centers = []

		for index in range(len(labels)):
			if labels[index] > 0:
				sensor_true_df.append(merged_series[index])
			else:
				sensor_false_df.append(merged_series[index])

		kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit(sensor_true_df)

		for center in kmeans.cluster_centers_:
			self.centers.append(center)


		kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=100, n_init=1, 
                               random_state = 42).fit(sensor_false_df)
		for center in kmeans.cluster_centers_:
			self.centers.append(center)


		true_instances = []
		false_instances = []
 
		self.sum_cluster = [0] * len(self.centers)
		self.count_cluster = [0] * len(self.centers)

		for index in range(len(labels)):
			cluster_id = self.find_closest_state(merged_series[index])

			self.sum_cluster[cluster_id] += labels[index]
			self.count_cluster[cluster_id] += 1


	def get_class_separation(self, cluster_id):

		if self.count_cluster[cluster_id] >= 4:
			return float(self.sum_cluster[cluster_id]) / max(1, self.count_cluster[cluster_id])

		return 0

	def find_closest_state(self, sensor_values):

		#sensor_values = self.scaler.transform([sensor_values])[0]

		min_index = -1
		min_distance = 999999999
		for center, index in zip(self.centers, range(len(self.centers))):

			net_sum = 0
			for dim_index in range(len(sensor_values)):
				net_sum += (sensor_values[dim_index] - center[dim_index]) * (sensor_values[dim_index] - center[dim_index])

			if net_sum < min_distance:
				min_index = index
				min_distance = net_sum

		return min_index















