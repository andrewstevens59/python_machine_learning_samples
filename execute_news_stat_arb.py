import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import sqlite3
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
import download_calendar as download_calendar
import string
import random as rand

import os
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from maximize_sharpe import *

from sklearn import mixture
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn import svm
from evaluate_model import evaluate
from uuid import getnode as get_mac
import socket
import paramiko
import json

import delta_process as delta_process
import breakout as breakout_process
import volatility_process as volatility_process
import barrier as barrier
import jump_process as jump_process
import gradient as gradient
import create_regimes as create_regimes
import logging
from close_trade import *
import os



class Order:

	def __init__(self):
		self.pair = ""
		self.dir = 0
		self.open_price = 0
		self.open_time = 0
		self.amount = 0
		self.PnL = 0
		self.tp_price = 0
		self.sl_price = 0
		self.actual_amount = 0
		self.account_number = None
		self.time_diff_hours = 0
		self.order_id = 0
		self.base_amount = 0
		self.sequence_id = 0
		self.model_key = None
		self.prediction_key = None
		self.margin_used = 0
		self.open_prediction = 0
		self.curr_prediction = 0
		self.portfolio_wt = 0
		self.calendar_time = 0

class TradeSeries:
	def __init__(self):
		self.series_open_price = 0
		self.anchor_price = 0
		self.max_delta = 0
		self.time_delta = 0
		
currency_pairs = [
    "AUD_CAD", "CHF_JPY", "EUR_NZD", "GBP_JPY",  
    "AUD_CHF", "EUR_AUD", "GBP_NZD", "USD_CAD", 
    "AUD_JPY", "EUR_CAD", "GBP_USD", "USD_CHF", 
    "AUD_NZD", "EUR_CHF", "EUR_USD", "NZD_CAD", 
    "AUD_USD", "EUR_GBP", "GBP_AUD", "NZD_CHF", 
    "CAD_CHF", "EUR_JPY", "GBP_CAD", "NZD_JPY", 
    "CAD_JPY", "GBP_CHF", "NZD_USD", "USD_JPY"
]

import datetime as dt

class MyFormatter(logging.Formatter):
    converter=dt.datetime.fromtimestamp
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s,%03d" % (t, record.msecs)
        return s

formatter = MyFormatter(fmt='%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_calendar_day(curr_date):

	pattern = re.compile("(1[012]|[1-9]):[0-5][0-9](\\s)?(?i)(am|pm)")

	from_zone = tz.tzlocal()
	to_zone = tz.tzutc()

	url='https://www.forexfactory.com/calendar.php?day=' + curr_date
	print url
	#Create a handle, page, to handle the contents of the website
	page = requests.get(url)
	#Store the contents of the website under doc
	doc = lh.fromstring(page.content)
	#Parse data that are stored between <tr>..</tr> of HTML
	tr_elements = doc.xpath('//tr')

	currencies = ["GBP", "USD", "AUD", "CAD", "NZD", "JPY", "CHF", "EUR"]

	calendar_data = []

	curr_time = None
	#Since out first row is the header, data is stored on the second row onwards
	for j in range(0,len(tr_elements)):
		#T is our j'th row
		T=tr_elements[j]

		found_currency = False
		found_description = False

		actual = None
		forecast = None
		previous = None
		space = None
		space1 = None
		currency = None
		description = None
		timestamp = None

		#Iterate through each element of the row
		for t in T.iterchildren():
			data=t.text_content().strip()

			if found_currency == True and space1 == None:
				space1 = data
				continue

			if found_currency == True:
				found_currency = False
				found_description = True
				description = data

				continue

			if found_description == True:

				if space == None:
					space = data
					print data, "Space"
					continue

				if actual == None:
					actual = data
					print data, "Actual"
					continue

				if forecast == None:
					forecast = data
					print data, "Forecast"
					continue

				if previous == None:
					previous = data
					print previous, "Previous"
					print description, "description"
		
					non_decimal = re.compile(r'[^\d.]+')

					try:
						actual = float(non_decimal.sub('', actual))
						forecast = float(non_decimal.sub('', forecast))
						previous = float(non_decimal.sub('', previous))
					except:
						actual = -1
						forecast = -1
						previous = -1


					calendar_data.append([curr_date, timestamp, currency, description, actual, forecast, previous]) 
					continue

			if data == "All Day":
				break

			if pattern.match(data):
				curr_time = data

				

			if data in currencies:
				print curr_date, curr_time, data
				found_currency = True
				currency = data

				local = datetime.datetime.strptime(curr_date + " " + curr_time, "%b%d.%Y %I:%M%p")

				local = local.replace(tzinfo=from_zone)

				# Convert time zone
				utc = local.astimezone(to_zone)

				timestamp = calendar.timegm(utc.timetuple())


	return calendar_data

def get_curr_calendar_day_features(description_map, pair):

	d = datetime.datetime.now() + datetime.timedelta(days=1)

	curr_date = datetime.datetime.now().strftime("%b%d.%Y").lower()

	week_day = datetime.datetime.now().weekday()

	print "curr day", week_day

	conn = sqlite3.connect('calendar_data.db')

	conn.execute('''
		CREATE TABLE IF NOT EXISTS CALENDAR_DATA
	         (
	         DATE           CHAR(50),
	         TIMESTAMP      LONG,
	         CURRENCY           CHAR(50),
	         DESCRIPTION 
	         ACTUAL           FLOAT,
	         PREVIOUS           FLOAT,
	         FORECAST           FLOAT,
	         CONSTRAINT UC_Person UNIQUE (DESCRIPTION,CURRENCY,TIMESTAMP)
	         );
	         ''')
	

	cursor.execute("SELECT TIMESTAMP FROM CALENDAR_DATA WHERE DATE=='" + curr_date + "'")
	calendar_data=cursor.fetchall()

	found_recent_news = False
	for row in calendar_data:
		if abs(time.time() - row[0]) < 1 * 60:
			print "find new news"
			found_recent_news = True

	for day_back in range(28):
		d = datetime.datetime.now() + datetime.timedelta(days=day_back)

		date = d.strftime("%b%d.%Y").lower()

		cursor.execute("SELECT * FROM CALENDAR_DATA WHERE DATE=='" + date + "'")
		data=cursor.fetchall()

		if len(data) == 0 or found_recent_news == True:

			calendar_data = get_calendar_day(date)

			for calendar_row in calendar_data:
				conn.execute('insert into CALENDAR_DATA values (?,?,?,?,?,?,?)', calendar_row)

	conn.commit()
	conn.close()

	global_feature_vector = []
	recent_news_time = 0
	for week_num in [2, 4, 8]:
		back_time_stamp = time.time() - (60 * 60 * 24 * 7 * week_num)
		
		cursor.execute("SELECT * FROM CALENDAR_DATA WHERE CURRENCY='" + pair[0:3] "' OR CURRENCY='" + pair[4:7] "' AND TIMESTAMP > '" + back_time_stamp + "' AND TIMESTAMP <= '" + time.time() + "'")
		calendar_data=cursor.fetchall()
 
		description_vector1_curr1 = [0] * len(description_map)
		description_vector2_curr1 = [0] * len(description_map)

		description_vector1_curr2 = [0] * len(description_map)
		description_vector2_curr2 = [0] * len(description_map)

		for row in calendar_data:
			recent_news_time = max(recent_news_time, row["TIMESTAMP"])
			
			if history["CURRENCY"] == pair[0:3]:
				description_vector1_curr1[description_map[history["DESCRIPTION"]]] = (history['ACTUAL'] - history['FORECAST'])
				description_vector2_curr1[description_map[history["DESCRIPTION"]]] = (history['ACTUAL'] - history['PREVIOUS'])
			else:
				description_vector1_curr2[description_map[history["DESCRIPTION"]]] = (history['ACTUAL'] - history['FORECAST'])
				description_vector2_curr2[description_map[history["DESCRIPTION"]]] = (history['ACTUAL'] - history['PREVIOUS'])

		global_feature_vector += description_vector1_curr1 + description_vector2_curr1 + description_vector1_curr2 + description_vector2_curr2
		
	return global_feature_vector, recent_news_time	

def get_open_trades(account_number, order_metadata, total_margin):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades?count=50"

	while next_link != None:
		response_value, header_value = sendCurlRequest(next_link, "GET")

		lines = header_value.split('\n')
		lines = filter(lambda line: line.startswith('Link:'), lines)

		if len(lines) > 0:
			line = lines[0]
			next_link = line[line.find("<") + 1 : line.find(">")]
		else:
			next_link = None

		j = json.loads(response_value)
		open_orders = j['trades']
		open_orders = sorted(open_orders, key=lambda order: order['openTime'])

		for trade in open_orders:
			s = trade['openTime'].replace(':', "-")
			s = s[0 : s.index('.')]
			order_id = trade['id']
			open_price = float(trade[u'price'])
			pair = trade[u'instrument']
			amount = float(trade[u'currentUnits'])
			pair = trade['instrument'].replace("/", "_")
			PnL = float(trade['unrealizedPL'])
			margin_used = float(trade['marginUsed'])


			pip_size = 0.0001
			if pair[4:7] == "JPY":
				pip_size = 0.01

			week_day_start = datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").weekday()

			time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())

			s = str(datetime.datetime.utcnow())
			s = s[0 : s.index('.')]

			week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

			time_end = time.time()

			time_diff_hours = time_end - time_start

			time_diff_hours /= 60 * 60

			week_num = int(time_diff_hours / (24 * 7))
			if week_num >= 1:
				time_diff_hours -= 48 * week_num

			if week_day_start != 6 and week_day_end < week_day_start:
				time_diff_hours -= 48
			elif week_day_end == 6 and week_day_end > week_day_start:
				time_diff_hours -= 48

			key = order_id + "_" + account_number
			
			if key not in order_metadata:
				order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
				trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
				print "not exist", key
				continue

			metadata = order_metadata[key]

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.time_diff_hours = time_diff_hours
			order.order_id = order_id
			order.account_number = account_number
			order.model_key = metadata["model_key"]
			order.open_prediction = metadata["open_prediction"]
			order.curr_prediction = metadata["curr_prediction"]
			order.dir = metadata["dir"]
			order.base_amount = metadata["base_amount"]
			order.sequence_id = metadata["sequence_id"]
			order.prediction_key = metadata["prediction_key"]
			order.calendar_time = metadata["calendar_time"]

			if 'portfolio_wt' in metadata:
				order.portfolio_wt = metadata["portfolio_wt"]

			order.margin_used = margin_used
			order.PnL = PnL

			if order.model_key not in white_list:
				order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
				trade_logger.info('Order Not In White List: ' + str(order_info)) 
				print "not white list", key
				continue

			orders.append(order)

	return orders, total_margin

def close_order(order, order_metadata, existing_order_amount, curr_price):

	key = order.order_id + "_" + order.account_number
	if key in order_metadata:
		del order_metadata[key]

	order_info, _ =  sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + order.account_number + "/trades/" + order.order_id + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info)) 
		trade_logger.info("Close Model Order: " + str({'model_key' : order.model_key, "Actual Profit" : str(order.PnL)})) 
		return order_metadata, existing_order_amount, True

	return order_metadata, existing_order_amount, False

def create_order(pair, curr_price, account_numbers, trade_dir, order_amount, existing_order_amount, base_model_key, stop_loss_price):

	if trade_dir == True:
		account_number = account_numbers[0]
	else:
		account_number = account_numbers[1]

	upper_inc = 0
	lower_inc = 0
	new_order_amount = order_amount
	fok_key = pair + "_" + account_number + "_" + str(order_amount)
	while fok_key in existing_order_amount:
		
		if rand.random() > 0.5 or lower_inc + 1 >= order_amount:
			upper_inc += 1
			new_order_amount = order_amount + upper_inc
			fok_key = pair + "_" + account_number + "_" + str(new_order_amount)
		else:
			lower_inc += 1
			new_order_amount = order_amount - lower_inc
			fok_key = pair + "_" + account_number + "_" + str(new_order_amount)

	existing_order_amount.add(fok_key)
	order_amount = new_order_amount


	precision = '%.4f'
	if pair[4:] == 'JPY':
		precision = '%.2f'

	if trade_dir == True:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % (curr_price + (20000 * pip_size))) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
		order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
	else:
		#tp_price = '"takeProfitOnFill": {"timeInForce": "GTC", "price": "' + str(precision % max(0, curr_price - (20000 * pip_size))) + '"}'
		sl_price = '"stopLossOnFill": {"timeInForce": "GTC", "price": "' + str(precision % stop_loss_price) + '"}'
		order_info, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {' + sl_price + ', "instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
		

	return order_info, account_number, order_amount

def update_orders(orders, group_metadata, order_metadata):

	sequence_id = 0
	for order in orders:

		order.sequence_id = sequence_id
		sequence_id += 1

		order_metadata[str(order.order_id) + "_" + order.account_number] = {
			"model_key" : order.model_key,
			"base_amount" : order.base_amount,
			"sequence_id" : order.sequence_id,
			"account_number" : order.account_number,
			"open_prediction" : order.open_prediction,
			"curr_prediction" : order.curr_prediction,
			"prediction_key" : order.prediction_key,
			"portfolio_wt" : order.portfolio_wt,
			"calendar_time" : order.calendar_time,
			"amount" : order.amount,
			"dir" : order.dir,
		}

def serialize_order(order):

	return {
			"model_key" : order.model_key,
			"base_amount" : order.base_amount,
			"sequence_id" : order.sequence_id,
			"account_number" : order.account_number,
			"open_prediction" : order.open_prediction,
			"curr_prediction" : order.curr_prediction,
			"prediction_key" : order.prediction_key,
			"portfolio_wt" : order.portfolio_wt,
			"calendar_time" : order.calendar_time,
			"amount" : order.amount,
			"dir" : order.dir,
		}


def close_group_trades(orders_by_model, model_key, is_force_close, order_metadata, group_metadata, total_margin_available, total_margin_used, existing_order_amount, curr_price, avg_prices, avg_spreads, total_models):

	if model_key not in orders_by_model:
		return

	orders = orders_by_model[model_key]

	total_pnl = 0
	for order in orders:
		total_pnl += order.PnL

	if model_key + "_min_profit" not in group_metadata:
		group_metadata[model_key + "_min_profit"] = total_pnl

	group_metadata[model_key + "_min_profit"] = min(total_pnl, group_metadata[model_key + "_min_profit"])

	trade_logger.info('Model: ' + model_key + \
			", Total Float Profit: " + str(total_pnl) + \
			", Reduce Amount: " + str(group_metadata[model_key + "_reduce_amount"]) + \
			", Total Orders: " + str(len(orders)))

	if (model_key + "_last_profit_udpate") not in group_metadata:
		group_metadata[model_key + "_last_profit_udpate"] = time.time()
		pickle.dump(group_metadata, open(root_dir + "group_metadata_news_release", 'wb'))

	group_metadata[model_key + "_last_profit_udpate"] = time.time()

	if model_key + "_reduce_amount" not in group_metadata:
		group_metadata[model_key + "_reduce_amount"] = len(orders)

	new_orders = []
	for order in orders:

		margin_used_factor = order.margin_used / (order.PnL + (total_margin_used / total_models))

		trade_logger.info(
			"    Pair: " + str(order.pair) + \
			", Float Profit: " + str(order.PnL) + \
			", Margin Used: " + str('%0.2f' % margin_used_factor) + \
			", Amount: " + str(order.amount) + \
			", Dir: " + str(order.dir) + \
			", Time Diff: " + str('%0.2f' % order.time_diff_hours))

		if (is_force_close or order.time_diff_hours >= 12) and group_metadata[order.pair + "_curr_spread"] < 5.0: 
			order_metadata, existing_order_amount, is_success = close_order(order, order_metadata, existing_order_amount, curr_price)
			if is_success == True:
				continue

		new_orders.append(order)

	group_metadata[model_key + "_reduce_amount"] = len(new_orders) + 1

	if len(new_orders) != len(orders):
		update_orders(new_orders, group_metadata, order_metadata)



def enter_group_trades(model_key, calendar_time, pair, prediction, order_metadata, group_metadata, orders_by_model, curr_model_wt, prev_model_wt, free_margin, used_margin, curr_price, account_numbers, existing_order_amount, curr_spread, avg_prices):

	base_model_key = model_key
	
	orders = []
	buy_count = 0
	sell_count = 0
	if base_model_key in orders_by_model:

		orders = orders_by_model[base_model_key]

		for order in orders:

			if order.calendar_time == calendar_time:
				# already have a matching order for this calendar time
				return True

			if order.dir:
				buy_count += 1
			else:
				sell_count += 1

			order.curr_prediction = prediction

		update_orders(orders, group_metadata, order_metadata)

	else:
		prev_model_wt[base_model_key] = curr_model_wt
		group_metadata[base_model_key + "_reduce_amount"] = 1
		group_metadata[base_model_key + "_prev_dir"] = None
		group_metadata[base_model_key + "_last_calendar_time"] = None

		if base_model_key + "_min_profit" in group_metadata:
			del group_metadata[base_model_key + "_min_profit"]

		if base_model_key + "_last_profit_udpate" in group_metadata:
			del group_metadata[base_model_key + "_last_profit_udpate"]

		if base_model_key in prev_model_wt:
			del prev_model_wt[base_model_key]

	if base_model_key + "_prev_dir" not in group_metadata:
		group_metadata[base_model_key + "_prev_dir"] = None

	if base_model_key + "_last_calendar_time" not in group_metadata:
		group_metadata[base_model_key + "_last_calendar_time"] = 0


	first_currency = pair[0:3]
	second_currency = pair[4:7]

	if second_currency != "USD":
		pair_mult = avg_prices[second_currency + "_USD"]
	else:
		pair_mult = 1.0

	new_order = Order()
	new_order.base_amount = curr_model_wt

	prices, times = get_time_series(pair, 24)
	kmeans = KMeans(n_clusters=min(10, len(prices)), init='k-means++', max_iter=100, n_init=1, 
                           random_state = 42).fit([[v] for v in prices])

	min_range = min([v[0] for v in kmeans.cluster_centers_.tolist()])
	max_range = max([v[0] for v in kmeans.cluster_centers_.tolist()])
	order_amount = int(round((new_order.base_amount * 0.02) / (pair_mult * abs(max_range - min_range))))

	if order_amount == 0:
		return False

	new_order.dir = (prediction > 0)

	if new_order.dir:
		stop_loss = min_range - abs(max_range - min_range)
	else:
		stop_loss = max_range + abs(max_range - min_range)

	order_info, account_number, order_amount = create_order(pair, curr_price, account_numbers, new_order.dir, order_amount, existing_order_amount, base_model_key, stop_loss)

	print str(order_info)

	order_info = json.loads(order_info)

	if 'orderFillTransaction' in order_info:
		trade_logger.info('New Order: ' + str(order_info)) 
		order_id = str(order_info['orderFillTransaction']['id'])
		group_metadata[base_model_key + "_last_profit_udpate"] = time.time()

		group_metadata[base_model_key + "_prev_dir"] = (prediction > 0)

		new_order.pair = pair
		new_order.order_id = order_id
		new_order.amount = order_amount
		new_order.account_number = account_number
		new_order.model_key = base_model_key
		new_order.calendar_time = calendar_time
		orders.append(new_order)

		trade_logger.info('Order MetaData: ' + str(serialize_order(new_order))) 

		update_orders(orders, group_metadata, order_metadata)

		pickle.dump(prev_model_wt, open(root_dir + "group_base_wt_news_release", 'wb'))
		pickle.dump(group_metadata, open(root_dir + "group_metadata_news_release", 'wb'))
		pickle.dump(order_metadata, open(root_dir + "order_metadata_news_release", 'wb'))

	else:
		trade_logger.info('Order Error: ' + str(order_info)) 

	return True

def get_time_series(symbol, time, granularity="H1"):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v1/candles?instrument=" + symbol + "&count=" + str(time) + "&candleFormat=midpoint&granularity=" + granularity + "&alignmentTimezone=America%2FNew_York")

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['candles']

	prices = []
	times = []

	index = 0
	while index < len(j):
		item = j[index]

		s = item['time']
		s = s[0 : s.index('.')]
		timestamp = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timetuple())

		times.append(timestamp)
		prices.append(item['closeMid'])
		index += 1

	return prices, times

def get_order_book(symbol, time, curr_price):

	response_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(pycurl.ENCODING, 'gzip') 
	curl.setopt(curl.URL, "https://api-fxtrade.oanda.com/v3/instruments/" + symbol + "/orderBook?time=" + time)

	curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer 2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8', 'Content-Type: application/json'])

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	j = json.loads(response_value)['orderBook']['buckets']

	features = []
	for item in j:

		if abs(float(item['price']) - curr_price) >  0:
			features.append([abs(float(item['price']) - curr_price), (float(item['longCountPercent']) - float(item['shortCountPercent'])) / abs(float(item['price']) - curr_price)])

	features = sorted(features, key=lambda x: abs(x[0]), reverse=True)

	features = features[:4]

	return [f[1] for f in features]

def calculate_time_diff(ts):

	date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

	week_day_start = date.weekday()

	time_start = calendar.timegm(date.timetuple())

	s = str(datetime.datetime.utcnow())
	s = s[0 : s.index('.')]

	week_day_end = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()

	time_end = time.time()

	time_diff_hours = time_end - time_start

	time_diff_hours /= 60 * 60

	week_num = int(time_diff_hours / (24 * 7))
	if week_num >= 1:
		time_diff_hours -= 48 * week_num

	if week_day_start != 6 and week_day_end < week_day_start:
		time_diff_hours -= 48
	elif week_day_end == 6 and week_day_end > week_day_start:
		time_diff_hours -= 48

	return time_diff_hours


def process_pending_trades(account_numbers, avg_spreads):

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	for account_number in account_numbers:
		response_value, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v3/accounts/" + account_number + "/summary", "GET")
		j = json.loads(response_value)

		account_profit = float(j['account'][u'unrealizedPL'])
		account_balance = float(j['account'][u'balance'])
		margin_available = float(j['account']['marginAvailable'])
		margin_used = float(j['account']['marginUsed'])

		total_balance += account_balance
		total_float_profit += account_profit
		total_margin_available += margin_available
		total_margin_used += margin_used

	trade_logger.info('Equity: ' + str(total_balance + total_float_profit))

	if os.path.isfile(root_dir + "group_metadata_news_release"):
		group_metadata = pickle.load(open(root_dir + "group_metadata_news_release", 'rb'))
	else:
		group_metadata = {}

	if os.path.isfile(root_dir + "order_metadata_news_release"):
		order_metadata = pickle.load(open(root_dir + "order_metadata_news_release", 'rb'))
	else:
		order_metadata = {}

	if os.path.isfile(root_dir + "group_base_wt_news_release"):
		prev_model_wt = pickle.load(open(root_dir + "group_base_wt_news_release", 'rb'))
	else:
		prev_model_wt = {}

	if "max_equity" not in group_metadata:
		group_metadata["max_equity"] = total_balance

	group_metadata["max_equity"] = max(group_metadata["max_equity"], total_balance)

	orders = []
	total_margin = 0
	for account_number in account_numbers:
		orders1, total_margin = get_open_trades(account_number, order_metadata, total_margin)
		orders += orders1

	orders_by_prediction = {}
	orders_by_model = {}
	last_order_time_diff_hours = {}
	existing_order_amount = set()

	for order in orders:

		if order.prediction_key not in orders_by_prediction:
			orders_by_prediction[order.prediction_key] = []

		if order.model_key not in orders_by_model:
			last_order_time_diff_hours[order.model_key] = 9999999999
			orders_by_model[order.model_key] = []

		if order.pair + "_" + order.account_number + "_" + str(order.amount) in existing_order_amount:
			trade_logger.info("Duplicate FIFO trade")

		existing_order_amount.add(order.pair + "_" + order.account_number + "_" + str(order.amount))
		orders_by_prediction[order.prediction_key].append(order)
		orders_by_model[order.model_key].append(order)

		last_order_time_diff_hours[order.model_key] = min(last_order_time_diff_hours[order.model_key], order.time_diff_hours)

	total_orders = len(orders)

	avg_prices = {}
	pair_bid_ask_map = {}
	for pair in currency_pairs:
		response, _ = sendCurlRequest("https://api-fxpractice.oanda.com/v1/prices?instruments=" + pair, "GET")
		response = json.loads(response)['prices']

		pip_size = 0.0001
		if pair[4:] == "JPY":
			pip_size = 0.01

		bid = None
		ask = None
		for spread_count in range(1):
			curr_spread = 0
			for price in response:
				if price['instrument'] == pair:
					curr_price = (price['bid'] + price['ask']) / 2
					curr_spread = abs(price['bid'] - price['ask']) / pip_size
					bid = price['bid']
					ask = price['ask']
					break

			if curr_price == 0:
				print "price not found"
				continue

		pair_bid_ask_map[pair] = {}
		pair_bid_ask_map[pair]['bid'] = bid
		pair_bid_ask_map[pair]['ask'] = ask

		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[first_currency + "_" + second_currency] = curr_price

	for pair in currency_pairs:
		first_currency = pair[0:3]
		second_currency = pair[4:7]
		avg_prices[second_currency + "_" + first_currency] = 1.0 / avg_prices[pair]

	base_calendar = get_curr_calendar_day()
	print base_calendar

	for pair in currency_pairs: 

		ideal_spread = avg_spreads[pair]
		pip_size = 0.0001
		if pair[4:] == "JPY":
			pip_size = 0.01

		actual_spread = abs(pair_bid_ask_map[pair]['bid'] - pair_bid_ask_map[pair]['ask']) / pip_size
		actual_spread /= ideal_spread 

		curr_price = abs(pair_bid_ask_map[pair]['bid'] + pair_bid_ask_map[pair]['ask']) / 2
		curr_spread = actual_spread

		last_processed_key = pair + "_last_processed"
		last_price_key = pair + "_last_price"
		if last_price_key not in group_metadata:
			group_metadata[last_price_key] = curr_price

		price_diff = abs(curr_price - group_metadata[last_price_key]) / pip_size
		price_diff /= 3
		curr_spread /= max(1, price_diff)

		print pair, "Curr Spread", curr_spread
		
		if curr_spread < 1.2:
			group_metadata[last_processed_key] = time.time()
			group_metadata[last_price_key] = curr_price

		group_metadata[pair + "_curr_spread"] = curr_spread
		
		if curr_spread > 1.3:
			continue

		first_currency = pair[0:3]
		second_currency = pair[4:7]
		calendar = base_calendar[(base_calendar["currency"] == first_currency) | (base_calendar["currency"] == second_currency)]

		stat_arb = pickle.load(open(root_dir + "trading_data/news_stat_arb_" + pair, 'rb'))
		description_map = stat_arb["description_map"]
		clf = stat_arb["clf"]

		feature_vector, news_time = get_curr_calendar_day_features(description_map, pair)

		time_diff_hours = calculate_time_diff(news_time)
		if time_diff_hours > 12:
			news_time = row["time"]
			model_key = pair + "_" + row["description"]

			if news_time > time.time():
				continue

			if row["description"] not in description_map:
				continue

			if row['actual'] < 0 or row['previous'] < 0 or row['forecast'] < 0:
				continue

			found_news_event = True
			time_diff_hours = calculate_time_diff(news_time)
			if time_diff_hours > 0.5:
				continue

			print "time_diff", time_diff_hours

			clf = description_map[row["description"]]["model"]
			feature_vector = [row['actual'] - row['forecast'], row['actual'] - row['previous']]

			delta = clf.predict([feature_vector])[0]

			print "Predict", prediction, "FV", global_feature_vector

			curr_model_wt = float(total_balance) * 0.02

			trade_logger.info('Model Delta: ' + str(model_key) + " : " + str(delta)) 

			base_model_key = enter_group_trades(model_key, news_time, pair, delta, order_metadata, \
				group_metadata, orders_by_model, curr_model_wt, prev_model_wt, \
				total_margin_available, total_margin_used, curr_price, \
				account_numbers, existing_order_amount, curr_spread, avg_prices)

			close_group_trades(orders_by_model, model_key, False, \
			 	order_metadata, group_metadata, total_margin_available, total_margin_used, \
			  	existing_order_amount, curr_price, avg_prices, avg_spreads, \
			  	len(orders_by_model))


	pickle.dump(prev_model_wt, open(root_dir + "group_base_wt_news_release", 'wb'))
	pickle.dump(group_metadata, open(root_dir + "group_metadata_news_release", 'wb'))
	pickle.dump(order_metadata, open(root_dir + "order_metadata_news_release", 'wb'))

	trade_logger.info('Float Profit: ' + str(total_float_profit)) 
	trade_logger.info('Total Orders: ' + str(total_orders)) 


accounts = [
	["101-011-9454699-002", "101-011-9454699-003"],
	["101-011-9454699-004", "101-011-9454699-005"],
	["101-011-9454699-006", "101-011-9454699-007"]
]


if get_mac() != 154505288144005:
	avg_spreads = pickle.load(open("/root/pair_avg_spread", 'rb'))
	root_dir = "/root/" 
else:
	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))
	root_dir = "/tmp/" 


	
trade_logger = setup_logger('first_logger', root_dir + "trade_news_release.log")

#process_pending_trades(["101-001-9145068-002", "101-001-9145068-003", "101-001-9145068-004", "101-001-9145068-005"]) #demo 2
process_pending_trades(accounts[1], avg_spreads) #demo
#process_pending_trades(["001-001-1370090-004", "001-001-1370090-003"])



if get_mac() == 154505288144005:

	print "Can't run locally - run only on remote server"

	root_dir = "/root/"
	
	import paramiko
	print "transferring"
	t = paramiko.Transport(("158.69.218.215", 22))
	t.connect(username="root", password="jEC1ZbfG")
	sftp = paramiko.SFTPClient.from_transport(t)
	sftp.put("/tmp/model_pair_wt", "/tmp/model_pair_wt")
	sftp.put("/tmp/pair_avg_spread", root_dir + "pair_avg_spread")
	sftp.put("/tmp/pair_avg_price", root_dir + "pair_avg_price")

	sftp.put("/tmp/pair_avg_spread", root_dir + "group_metadata_news_release")
	sftp.put("/tmp/pair_avg_price", root_dir + "order_metadata_news_release")

	avg_spreads = pickle.load(open("/tmp/pair_avg_spread", 'rb'))



