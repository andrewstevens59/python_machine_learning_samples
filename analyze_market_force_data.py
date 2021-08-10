import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
from pytz import timezone
import xgboost as xgb
from datetime import timedelta
import mysql.connector
from lxml.html import fromstring
from itertools import cycle
from scipy import stats
import datetime as dt
import traceback
import json

import re

import matplotlib

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import json
import copy

import math
import sys
import re

import numpy as np
import pandas as pd 
import pycurl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket
import paramiko
import json


import os
import bisect

import paramiko
import json

import logging
import os
import enum

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import logging
import datetime as dt
from uuid import getnode as get_mac
import traceback

def load_time_series(symbol, year, is_bid_file):

	if get_mac() == 150538578859218:
		prefix = '/Users/andrewstevens/Downloads/economic_calendar/'
	else:
		prefix = '/root/trading_data/'

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir(prefix) if isfile(join(prefix, f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if pair in file and 'Candlestick_1_Hour_BID' in file:
			break

	if pair not in file:
		return None

	with open(prefix + file) as f:
		content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	from_zone = tz.gettz('America/New_York')
	to_zone = tz.tzutc()

	prices = []
	times = []
	volumes = []

	content = content[1:]

	if year != None:
		start_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".1.1 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())
		end_time = calendar.timegm(datetime.datetime.strptime(str(year) + ".12.31 00:00:00", "%Y.%m.%d %H:%M:%S").timetuple())

	for index in range(len(content)):

		toks = content[index].split(',')
		utc = datetime.datetime.strptime(toks[0], "%d.%m.%Y %H:%M:%S.%f")

		time = calendar.timegm(utc.timetuple())

		if year == None or (time >= start_time and time < end_time):

			high = float(toks[2])
			low = float(toks[3])
			o_price = float(toks[1])
			c_price = float(toks[4])
			volume = float(toks[5])

			if high != low or volume > 0:
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes

def get_calendar_df(pair, year): 

	if pair != None:
		currencies = [pair[0:3], pair[4:7]]
	else:
		currencies = None

	if get_mac() == 150538578859218:
		with open("/Users/andrewstevens/Downloads/economic_calendar/calendar_" + str(year) + ".txt") as f:
			content = f.readlines()
	else:
		with open("/root/trading_data/calendar_" + str(year) + ".txt") as f:
			content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	lines = [x.strip() for x in content] 

	from_zone = tz.gettz('US/Eastern')
	to_zone = tz.gettz('UTC')

	contents = []

	for line in lines:
		line = line[len("2018-12-23 22:44:55 "):]
		toks = line.split(",")

		if currencies == None or toks[1] in currencies:

			time = int(toks[0])
			actual = unicode(toks[3], 'utf-8')
			forecast = unicode(toks[4], 'utf-8')
			previous = unicode(toks[5], 'utf-8')

			actual = "".join([v for v in actual if v.isnumeric() or v in ['.', '+', '-']])
			if len(actual) == 0:
				continue

			try:
				actual = float(actual)

				forecast = "".join([v for v in forecast if v.isnumeric() or v in ['.', '+', '-']])
				if len(forecast) > 0:
					forecast = float(forecast)
				else:
					forecast = actual

				previous = "".join([v for v in previous if v.isnumeric() or v in ['.', '+', '-']])
				if len(previous) > 0:
					previous = float(previous)
				else:
					previous = actual

				contents.append([toks[1], time, toks[2], actual, forecast, previous, int(toks[6]), toks[7]])
			except:
				pass

	return pd.DataFrame(contents, columns=["currency", "time", "description", "actual", "forecast", "previous", "impact", "better_worse"])

news_release_stat_df = pd.read_csv("news_dist_stats.csv")
news_release_stat_df.set_index("key", inplace=True)

select_pair = "AUD_CAD"
df = pd.read_excel("/Users/andrewstevens/Downloads/AUD_Crosses_MarketForceData.xlsx", 'AUDCAD')
print (df)

from_zone = tz.gettz('US/Eastern')
to_zone = tz.gettz('UTC')

#gmt = est - 5
df["Date"] = df["Date"].apply(lambda x: calendar.timegm(datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S").timetuple()))
df["Date"] = df["Date"].apply(lambda x: x + (5 * 60 * 60))
df["Diff"] = df.apply(lambda x: x["VOL_Long%"] - x["Vol_Short%"], axis=1)
dates = df["Date"].values.tolist()

df_set = []
for year in range(2007, 2020):
	print (year, len(get_calendar_df(None, year)))
	df_set.append(get_calendar_df(None, year))

calendar_df = pd.concat(df_set)

prices, times, volumes = load_time_series("AUD_CAD", None, False)
price_df = pd.DataFrame()
price_df["prices"] = prices
price_df["times"] = times

for index in range(0, len(dates), 25):

	plt.title(select_pair + " Basket Correlation Plot")

	start_timestamp = dates[index]
	end_timestamp = dates[index+250]

	sub_df = price_df[(price_df["times"] >= start_timestamp) & (price_df["times"] <= end_timestamp)]
	times = sub_df["times"].values.tolist()
	prices = sub_df["prices"].values.tolist()
	#plt.plot([bisect.bisect(times, timestamp) for timestamp in times], prices, label="Time Series")
	
	sub_df = df[(df["Date"] >= start_timestamp) & (df["Date"] <= end_timestamp)]
	dates = sub_df["Date"].values.tolist()
	diffs = sub_df["Diff"].values.tolist()
	print (diffs)

	plt.plot([bisect.bisect(times, timestamp) for timestamp in dates], diffs, label="Market Force BUY - SELL")
		
	sub_df = calendar_df[(calendar_df["currency"] == "AUD") | (calendar_df["currency"] == "CAD")]
	sub_df = sub_df[(sub_df["time"] >= start_timestamp) & (sub_df["time"] <= end_timestamp)]
	timestamps = []
	z_scores = []
	impacts = []
	impact_set = set()

	for index, row in sub_df.iterrows():
		key = row["description"] + "_" + row["currency"]
		stat_row = news_release_stat_df[news_release_stat_df.index == key]
		if len(stat_row) == 0:
			continue

		stat_row = stat_row.iloc[0]

		sign = stat_row["sign"]

		if stat_row["forecast_std"] > 0:
			z_score1 = (float(row["actual"] - row["forecast"]) - stat_row["forecast_mean"]) / stat_row["forecast_std"]
		else:
			z_score1 = None

		if z_score1 is not None:
			timestamps.append(row["time"])
			z_scores.append(z_score1)
			impacts.append(row["impact"])
			impact_set.add(row["impact"])

	for impact in impact_set:
		select_z_scores = [abs(z) / 10 for z, i in zip(z_scores, impacts) if i == impact]
		select_timestamps = [t for t, i in zip(timestamps, impacts) if i == impact]

		if impact == 1:
			impact_label = "Low"
			color_code = "darkseagreen"
		elif impact == 2:
			impact_label = "Medium"
			color_code = "palegreen"
		elif impact == 3:
			impact_label = "High"
			color_code = "lime"


		plt.plot([bisect.bisect(times, timestamp) for timestamp in select_timestamps], select_z_scores, 'o', color=color_code, label="Economic News Impact {}".format(impact_label))

	'''
	selectes_times = []
	x_tick_indexes = []
	prev_time = None
	date_range = [[datetime.datetime.utcfromtimestamp(time_subset[t]).strftime('%m-%d'), bisect.bisect(time_subset, time_subset[t])] for t in range(len(time_subset))]
	for item in date_range:

		if len(x_tick_indexes) > 0 and abs(item[1] - x_tick_indexes[-1]) < 24:
			continue

		if item[0] != prev_time:
			selectes_times.append(item[0])
			x_tick_indexes.append(item[1])
			prev_time = item[0]
	'''

	plt.ylabel("Correlation")
	#plt.xticks(x_tick_indexes, selectes_times, rotation=30)

	plt.legend()
	#plt.savefig("/var/www/html/images/{}_news_correlation.png".format(select_currency))
	#pdf.savefig()
	plt.show()



