import pandas as pd
import numpy as np
import sys
import math
from datetime import datetime
from random import *
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import time
import os.path

from sklearn import datasets, linear_model


from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import numpy as np
from sklearn.decomposition import PCA

import pickle

class UserLocation:

	# This was found on Google - I hope this is okay
	def haversine(self, lat1, lon1, lat2, lon2):
		
		"""
	    Calculate the great circle distance between two points
	    on the earth (specified in decimal degrees)

	    All args must be of equal length.    

	    """
		lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

		dlon = lon2 - lon1
		dlat = lat2 - lat1

		a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

		c = 2 * np.arcsin(np.sqrt(a))
		km = 6367 * c
		return km


	def findStops(self, path):

		df = pd.read_csv(path, delimiter='\t')

		df['dist'] = self.haversine(df['lng'].shift(), df['lat'].shift(), df.loc[1:, 'lng'], df.loc[1:, 'lat'])


		min_dist = df['dist'].dropna().quantile(0.2)

		threshold_accuracy = df['accuracy'].dropna().quantile(0.7)

		stop_times = []

		start_time = 0
		end_time = 0

		count = 0
		stop_entry = []
		for index, row in df.iterrows():

			if row['accuracy'] > threshold_accuracy:
				# noisy estimate remove it
				continue

			if start_time == 0:
				start_time = row['time']

			end_time = row['time']

			count = count + 1

			if row['dist'] > min_dist:
				if count > 1: 
					stop_times.append(stop_entry)

				start_time = 0
				count = 0

			stop_entry = [row['installid'], start_time, end_time, row['lat'], row['lng']]
    	
		if count > 1: 
			stop_times.append(stop_entry)

		return pd.DataFrame.from_records(stop_times, columns = ["installid", "start_time", "end_time", "lat", "lng"])


stops = UserLocation()

locations = stops.findStops("TakeHomeData.txt")

locations.to_csv("stopping_data.csv", index=False)

print locations[['lat', 'lng', 'start_time', 'end_time']]

plt.plot(locations['lat'], locations['lng'], '-o')
plt.title('Stop Locations')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()



