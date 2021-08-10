import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from sklearn.cluster import KMeans
from numpy import linalg as LA

import time
import datetime
import calendar
from dateutil import tz
import requests
import json
import copy

import math
import sys
import re

import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import string
import random as rand

from uuid import getnode as get_mac
import socket


import os
import mysql.connector

import logging
import os


def load_time_series(symbol, year, is_bid_file):

	if get_mac() == 154505288144005:
		prefix = '/Users/andrewstevens/Downloads/economic_calendar/'
	else:
		prefix = '/root/trading_data/'

	from os import listdir
	from os.path import isfile, join
	onlyfiles = [f for f in listdir(prefix) if isfile(join(prefix, f))]

	pair = symbol[0:3] + symbol[4:7]

	for file in onlyfiles:

		if is_bid_file and pair in file and 'Candlestick_1_Hour_BID' in file:
			break

		if is_bid_file == False and pair in file and 'Candlestick_1_Hour_ASK' in file:
			break

	with open(prefix + file) as f:
	    content = f.readlines()

	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	from_zone = tz.tzlocal()
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

		s = toks[0][:-len(".000 GMT-0500")]
		local = datetime.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")

		local = local.replace(tzinfo=from_zone)

		# Convert time zone
		utc = local.astimezone(to_zone)

		time = calendar.timegm(utc.timetuple())

		if year == None or (time >= start_time and time < end_time):

			high = float(toks[2])
			low = float(toks[3])
			o_price = float(toks[1])
			c_price = float(toks[4])
			volume = float(toks[5])

			if high != low:
				prices.append(c_price)
				times.append(time)
				volumes.append(volume)

	return prices, times, volumes

currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]



cnx = mysql.connector.connect(user='andstv48', password='Password81',
                              host='66.33.203.187',
                              database='newscapital')



for currency_pair in currency_pairs:

	curr_time = 0
	cursor = cnx.cursor()
	query = ("SELECT max(timestamp) FROM historic_exchange_rates where \
						currency_pair = '" + currency_pair + "' \
						")

	cursor.execute(query)

	setup_rows = []
	for row1 in cursor:
		setup_rows.append(row1)

	cursor.close()

	if setup_rows[0][0] != None:
		curr_time = setup_rows[0][0]

	prices, times, volumes = load_time_series(currency_pair, None, True)
	buy_price_df = pd.DataFrame()
	buy_price_df['times'] = times
	buy_price_df["price_buy"] = prices
	buy_price_df["volume_buy"] = volumes
	buy_price_df.set_index('times', inplace=True)
	buy_price_df.fillna(method='ffill', inplace=True)

	prices, times, volumes = load_time_series(currency_pair, None, False)
	sell_price_df = pd.DataFrame()
	sell_price_df['times'] = times
	sell_price_df["price_sell"] = prices
	sell_price_df["volume_sell"] = volumes
	sell_price_df.set_index('times', inplace=True)
	sell_price_df.fillna(method='ffill', inplace=True)

	price_df = buy_price_df.join(sell_price_df)
	price_df.reset_index(inplace=True)

	for index, row in price_df.iterrows():

		if row['times'] <= curr_time:
			continue

		if True:
			cursor = cnx.cursor()
			query = ("INSERT INTO historic_exchange_rates values ( \
				'" + currency_pair + "', \
				'" + str(row["times"]) + "', \
				'" + str(row["price_buy"]) + "', \
				'" + str(row["price_sell"]) + "' \
				)")
			cursor.execute(query)
			cnx.commit()
			cursor.close()
		

