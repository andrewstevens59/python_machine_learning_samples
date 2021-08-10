import sys
import math
from datetime import datetime
from random import *
import os.path

import pickle

from StringIO import StringIO
from sklearn.cluster import KMeans
from numpy import linalg as LA
import mysql.connector

import time
import datetime
import calendar
from dateutil import tz
import requests
import lxml.html as lh
import json
import copy
import psutil

import socket
import sys
import time
from bisect import bisect

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
import gzip, cPickle
import string
import random as rand

from os import listdir
from os.path import isfile, join

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

api_key = None
account_type = "fxtrade"

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
		self.barrier_size = 0
		self.carry_cost = 0
		self.ideal_price = 0
		

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

def is_valid_trading_period(ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    if s.weekday() in {4} and s.hour in {21, 22, 23}:
        return False
    if s.weekday() in {5}:
        return False
    if s.weekday() in {6} and s.hour < 21:
        return False
    
    return True

def sendCurlRequest(url, request_type, post_data = None):
	response_buffer = StringIO()
	header_buffer = StringIO()
	curl = pycurl.Curl()

	curl.setopt(curl.URL, url)

	curl.setopt(pycurl.CUSTOMREQUEST, request_type)

	curl.setopt(curl.WRITEFUNCTION, response_buffer.write)

	curl.setopt(curl.HEADERFUNCTION, header_buffer.write)

	#2e076e68f33a65ce407a4f094accbee3-6d9a9cf8bcc0a6ea0d26bfe2a85341a8 actual

	#3168adee62ceb8b5f750efcec83c2509-1db7870026de5cccb33b220c100a07ab demo

	#16736a148307bf5d91f9f03dd7c91623-af6a25cf6249c9083f11593ad3899f89 demo2

	print url, api_key


	if post_data != None:
		print post_data
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer ' + api_key, 'Content-Type: application/json'])
		curl.setopt(pycurl.POSTFIELDS, post_data)
	else:
		curl.setopt(pycurl.HTTPHEADER, ['Authorization: Bearer ' + api_key])

	curl.perform()
	curl.close()

	response_value = response_buffer.getvalue()
	header_value = header_buffer.getvalue()

	return response_value, header_value

def setup_logger(name, log_file, level=logging.INFO):
	"""Function setup as many loggers as you want"""

	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger

def calculate_time_diff(now_time, ts):

    date = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    s = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    date = datetime.datetime.utcfromtimestamp(now_time).strftime('%Y-%m-%d %H:%M:%S')
    e = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")

    _diff = (e - s)

    while s < e:
        max_hour = 24
        if s.day == e.day:
            max_hour = e.hour

        if s.weekday() in {4}:
            max_hour = 21

        if s.weekday() in {4} and s.hour in {21, 22, 23}:
            hours = 1
            _diff -= timedelta(hours=hours)
        elif s.weekday() in {5}:
            hours = max_hour - s.hour
            _diff -= timedelta(hours=hours)
        elif s.weekday() in {6} and s.hour < 21:
            hours = min(21, max_hour) - s.hour
            _diff -= timedelta(hours=hours)
        else:
            hours = max_hour - s.hour

        if hours == 0:
            break
        s += timedelta(hours=hours)

    return (_diff.total_seconds() / (60 * 60))

def get_open_trades(account_number, order_metadata, total_margin):

	orders = []
	pair_time_diff = {}
	next_link = "https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades?count=50"

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

			time_start = calendar.timegm(datetime.datetime.strptime(s, "%Y-%m-%dT%H-%M-%S").timetuple())
			time_diff = calculate_time_diff(time.time(), time_start)

			'''
			if time_diff > 24 * 10:
				order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/trades/" + order_id + "/close", "PUT")
				trade_logger.info('Close Not Exist Order: ' + str(order_info)) 
				print "not exist", key, order_metadata
				continue
			'''

			order = Order()
			order.open_price = open_price
			order.amount = abs(amount)
			order.pair = pair
			order.dir = amount > 0
			order.order_id = int(order_id)
			order.account_number = account_number
			order.open_time = time_start
			order.margin_used = margin_used
			order.PnL = PnL

			orders.append(order)

	return orders, total_margin

def close_order(order):

	order_info, _ =  sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + order.account_number + "/trades/" + str(order.order_id) + "/close", "PUT")

	if "ORDER_CANCEL" not in order_info:
		trade_logger.info('Close Order: ' + str(order_info))  
		return True

	return False


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

def create_order(pair, account_numbers, trade_dir, order_amount):


    account_number = account_numbers[0]
 

    precision = '%.4f'
    if pair[4:7] == 'JPY':
        precision = '%.2f'

    if trade_dir == True:
        order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": ' + str(order_amount) + ', "type": "MARKET", "side" : "buy"}}')
    else:
        order_info, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/orders", "POST", '{ "order" : {"instrument": "' + pair + '", "units": ' + str(-order_amount) + ', "type": "MARKET", "side" : "sell"}}')
        

    return order_info

def process_pending_trades(account_numbers, existing_orders, existing_time_frames):

	total_balance = 0
	total_float_profit = 0
	total_margin_available = 0
	total_margin_used = 0
	for account_number in account_numbers:
		response_value, _ = sendCurlRequest("https://api-" + account_type + ".oanda.com/v3/accounts/" + account_number + "/summary", "GET")
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

	pair_map = pickle.load(open("{}price_deltas.pickle".format(root_dir), "rb"))


	orders = []
	for account_number in account_numbers:
		orders1, total_margin = get_open_trades(account_number, {}, total_margin_used)
		orders += orders1

	for order in orders:
		if str(order.order_id) not in existing_orders:

			percentiles = pair_map[order.pair]["percentiles"]
			deltas = pair_map[order.pair]["deltas"]

			max_percentile = 0
			time_frame = []
			for key in percentiles:
				if key > 20:
					continue

				if str(key) + order.pair + str(order.dir) not in existing_time_frames:
					if percentiles[key] > max_percentile and (deltas[key] > 0) == order.dir:
						time_frame = key
						max_percentile = percentiles[key]
				else:
					print ("found")

			cursor = cnx.cursor()
			query = ("insert into user_trade_positions (user_id, order_id, time_frame, dir, pair) values (61, '{}', '{}', '{}', '{}')".format(order.order_id, time_frame, 1 if order.dir else 0, order.pair))
			cursor.execute(query)
			cnx.commit()

		else:
			print ("existing order")


			

def checkIfProcessRunning(processName, command):
    count = 0
    #Iterate over the all the running process
    for proc in psutil.process_iter():

        try:
            cmdline = proc.cmdline()

            # Check if process name contains the given name string.
            if len(cmdline) > 3 and processName.lower() in cmdline[2] and command == cmdline[3]:
                count += 1
            elif len(cmdline) > 2 and processName.lower() in cmdline[1] and command == cmdline[2]: 
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if count >= 3:
        sys.exit(0)

checkIfProcessRunning('execute_equity_curve_z_score.py', '')


if get_mac() != 150538578859218:
	root_dir = "/root/" 
else:
	root_dir = "/tmp/" 






trade_logger = setup_logger('first_logger', root_dir + "equity_z_score.log")


cnx = mysql.connector.connect(user='andstv48', password='Password81',
							  host='mysql.newscaptial.com',
							  database='newscapital')

cursor = cnx.cursor()
query = ("SELECT order_id, pair, time_frame, dir FROM user_trade_positions t2 where user_id=61")

cursor.execute(query)

existing_orders = []
existing_time_frames = []
for row in cursor:
	order_id = str(row[0])
	pair = row[1]
	time_frame = row[2]
	trade_dir = row[3]
	existing_orders.append(order_id)
	existing_time_frames.append(str(time_frame) + pair + str(trade_dir))

cursor.close()

cursor = cnx.cursor()
query = ("SELECT account_nbr, api_key FROM managed_accounts t2 where t2.user_id=61 and t2.account_nbr='001-011-2949857-007'")

cursor.execute(query)

setup_rows = []
for row1 in cursor:
	setup_rows.append(row1)

cursor.close()

for row in setup_rows:

	account_nbr = row[0]
	api_key = row[1]

	process_pending_trades([account_nbr], existing_orders, existing_time_frames) #demo


