

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

from sklearn import datasets, linear_model, metrics

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

import numpy as np
from sklearn.decomposition import PCA

import pickle

## Structure of this code could be improved - but this was not the main focus of this exercise.

def model_station_rate(trip_df, station_id, station_type):

	trip_df['is_station'] = (trip_df[station_type] == station_id).astype(int)

	# find the rate (fraction) of bikes leaving or arriving for a given station id for each given date
	labels = trip_df.groupby(['Join Date']).apply(lambda x: float(x['is_station'].sum()) / max(x['is_station'].count(), 1)).reset_index(name='rate')

	join_df = pd.merge(trip_df, labels, on='Join Date')

	labels = join_df['rate']

	join_df = join_df[[u'Max TemperatureF', u'Mean TemperatureF', u'Min TemperatureF',
       u'Max Dew PointF', u'MeanDew PointF', u'Min DewpointF', u'Max Humidity',
       u'Mean Humidity', u'Min Humidity', u'Max Sea Level PressureIn',
       u'Mean Sea Level PressureIn', u'Min Sea Level PressureIn',
       u'Max VisibilityMiles', u'Mean VisibilityMiles', u'Min VisibilityMiles',
       u'Max Wind SpeedMPH', u'Mean Wind SpeedMPH', u'Max Gust SpeedMPH',
       u'PrecipitationIn', u'CloudCover', u'Events', u'WindDirDegrees']]


	train = pd.get_dummies(join_df, columns = ['Events'])

	train_sample_num = int(len(train) * 0.75)

	# We don't use train_test_split here as this function randomly samples a train and test split 
	# This is not appropriate we need to respect the order of the training samples w.r.t date

	X_train = train.iloc[0:train_sample_num]
	Y_train = labels.iloc[0:train_sample_num]

	X_test = train.iloc[train_sample_num:]
	Y_test = labels.iloc[train_sample_num:]

	regression = linear_model.LinearRegression()
	regression.fit(X_train, Y_train)

	print "Columns", join_df.columns
	print "Regression Coeff", regression.coef_

	predictions = regression.predict(X_test)

	print "RMSE", math.sqrt(metrics.mean_squared_error(Y_test, predictions))
	print "Mean Rate", Y_train.mean()

	return regression, X_test.iloc[-1]



def find_stock_change_rate(station_id):

	weather_df = pd.read_csv("weather_data.csv", delimiter=',')
	trip_df = pd.read_csv("trip_data.csv", delimiter=',')
	station_df = pd.read_csv("station_data.csv", delimiter=',')

	weather_df.fillna(weather_df.mean(), inplace=True)

	trip_df['Join Date'] = trip_df['Start Date'].str.slice(0, 10)
	weather_df['Join Date'] = weather_df['Date'].str.slice(0, 10)

	station_df['Start Station'] = station_df['Id']
	station_df['End Station'] = station_df['Id']

	join_df = trip_df.merge(station_df, on='Start Station')
	join_df = join_df.merge(weather_df, on='Join Date')


	leave_rate_model, last_observation = model_station_rate(join_df, station_id, 'Start Station')

	join_df = trip_df.merge(station_df, on='End Station')
	join_df = join_df.merge(weather_df, on='Join Date')

	arrive_rate_model, last_observation = model_station_rate(join_df, station_id, 'End Station')


	return (arrive_rate_model.predict(last_observation) - leave_rate_model.predict(last_observation)) / 24


# just choose an arbitrary station id to test it
accumulation_rate = find_stock_change_rate(50)

print "accumulation rate for station 50", accumulation_rate






